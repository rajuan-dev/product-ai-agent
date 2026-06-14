from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from statistics import mean, median
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import urlopen
from uuid import uuid4

from fastapi import HTTPException, status

from app.auth import AuthenticatedUser
from app.agents.amazon_agent import AmazonAgent
from app.agents.ebay_agent import EbayAgent
from app.agents.etsy_agent import EtsyAgent
from app.agents.image_agent import ImageAgent
from app.agents.product_optimization_agent import ProductOptimizationAgent
from app.agents.shopify_agent import ShopifyAgent
from app.agents.tiktok_agent import TikTokAgent
from app.config import get_settings
from app.orchestrator.pipeline import ProductPipeline
from app.schemas.request import (
    ProductOptimizationRequest,
    ProductUpdateRequest,
    VariantCreateRequest,
)
from app.schemas.response import (
    AmazonResponse,
    CoreProductResponse,
    EbayResponse,
    EtsyResponse,
    GeneratedImagesResponse,
    ImageValidationResponse,
    ImageVariantResponse,
    IntelligenceLayerResponse,
    MarketplaceLiteral,
    MarketplacePricingSnapshotResponse,
    MarketplaceResearchResponse,
    MarketResearchBundleResponse,
    MarketplaceVariantsResponse,
    PaginatedProductListResponse,
    PipelineValidationResponse,
    ProductPricingSnapshotResponse,
    DynamicPricingQueryResponse,
    ProductListItemResponse,
    ProductPipelineResponse,
    ProductRecordResponse,
    ProductVariantResponse,
    ResearchEvidenceResponse,
    SectionValidationResponse,
    SeoInsightsResponse,
    SuggestedPriceRangeResponse,
    ShopifyResponse,
    TikTokResponse,
)
from app.services.gemini_service import GeminiServiceError
from app.services.image_service import ImagePayload
from app.services.mongo_product_store import MongoProductStore
from app.services.output_service import OutputService
from app.services.product_store import ProductStore
from app.services.s3_service import S3Service
from app.utils.prompts import PromptRegistry
from app.utils.product_text import title_keywords, unique_strings


class ProductService:
    _GEMINI_PRICING_SCHEMA = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "insufficient_data"]},
            "minimum": {"type": "number"},
            "maximum": {"type": "number"},
            "recommended": {"type": "number"},
            "currency": {"type": "string"},
            "market_signal": {"type": "string"},
            "analysis_summary": {"type": "string"},
            "search_queries": {"type": "array", "items": {"type": "string"}},
            "comparable_count": {"type": "integer"},
            "price_sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "price": {"type": "number"},
                        "currency": {"type": "string"},
                    },
                    "required": ["source", "title"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["status"],
        "additionalProperties": False,
    }

    _shared_settings = None
    _shared_pipeline = None
    _shared_store = None
    _shared_output_service = None
    _shared_image_agent = None
    _shared_amazon_agent = None
    _shared_ebay_agent = None
    _shared_etsy_agent = None
    _shared_tiktok_agent = None
    _shared_shopify_agent = None
    _shared_optimization_agent = None

    def __init__(self) -> None:
        settings = self._shared_settings or get_settings()
        self.__class__._shared_settings = settings
        self.settings = settings
        self.pipeline = self._shared_pipeline or ProductPipeline()
        self.__class__._shared_pipeline = self.pipeline
        self.store = self._shared_store or (
            MongoProductStore(
                mongodb_uri=settings.mongodb_uri,
                db_name=settings.mongodb_db_name,
                products_collection=settings.mongodb_products_collection,
                imports_collection=settings.mongodb_imports_collection,
            )
            if settings.mongodb_uri and settings.mongodb_enabled
            else ProductStore(settings.output_dir)
        )
        self.__class__._shared_store = self.store
        self.output_service = self._shared_output_service or OutputService(
            settings.output_dir,
            s3_service=S3Service(
                region=settings.aws_region,
                bucket_name=settings.aws_s3_bucket,
                access_key_id=settings.aws_access_key_id,
                secret_access_key=settings.aws_secret_access_key,
                prefix=settings.aws_s3_prefix,
            ),
            local_output_enabled=settings.local_output_enabled,
        )
        self.__class__._shared_output_service = self.output_service
        self.image_agent = self._shared_image_agent or ImageAgent(self.pipeline.openai_service)
        self.__class__._shared_image_agent = self.image_agent
        self.amazon_agent = self._shared_amazon_agent or AmazonAgent(self.pipeline.openai_service, self.pipeline.gemini_service)
        self.__class__._shared_amazon_agent = self.amazon_agent
        self.ebay_agent = self._shared_ebay_agent or EbayAgent(self.pipeline.openai_service, self.pipeline.gemini_service)
        self.__class__._shared_ebay_agent = self.ebay_agent
        self.etsy_agent = self._shared_etsy_agent or EtsyAgent(self.pipeline.openai_service, self.pipeline.gemini_service)
        self.__class__._shared_etsy_agent = self.etsy_agent
        self.tiktok_agent = self._shared_tiktok_agent or TikTokAgent(self.pipeline.openai_service, self.pipeline.gemini_service)
        self.__class__._shared_tiktok_agent = self.tiktok_agent
        self.shopify_agent = self._shared_shopify_agent or ShopifyAgent(self.pipeline.openai_service, self.pipeline.gemini_service)
        self.__class__._shared_shopify_agent = self.shopify_agent
        self.optimization_agent = self._shared_optimization_agent or ProductOptimizationAgent(self.pipeline.openai_service, self.pipeline.gemini_service)
        self.__class__._shared_optimization_agent = self.optimization_agent

    @classmethod
    def reset_shared_state(cls) -> None:
        store = cls._shared_store
        close = getattr(store, "close", None)
        if callable(close):
            close()

        cls._shared_settings = None
        cls._shared_pipeline = None
        cls._shared_store = None
        cls._shared_output_service = None
        cls._shared_image_agent = None
        cls._shared_amazon_agent = None
        cls._shared_ebay_agent = None
        cls._shared_etsy_agent = None
        cls._shared_tiktok_agent = None
        cls._shared_shopify_agent = None
        cls._shared_optimization_agent = None

    async def generate_text_only(
        self,
        image: ImagePayload,
        title: str,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        result = await self._build_generated_product(
            image=image,
            title=title,
            image_marketplaces=[],
        )
        record = ProductRecordResponse(
            id=self._new_id(),
            status="draft",
            created_at=self._timestamp(),
            updated_at=self._timestamp(),
            run_id=result["run_id"],
            product=result["product"],
            variants=MarketplaceVariantsResponse(),
        )
        self.store.save(record, user_id=current_user.user_id if current_user is not None else None)
        return record

    async def generate_marketplace_draft(
        self,
        image: ImagePayload,
        title: str,
        marketplace: MarketplaceLiteral,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        result = await self._build_generated_product(
            image=image,
            title=title,
            image_marketplaces=[marketplace],
        )
        record = ProductRecordResponse(
            id=self._new_id(),
            status="draft",
            created_at=self._timestamp(),
            updated_at=self._timestamp(),
            run_id=result["run_id"],
            product=result["product"],
            variants=MarketplaceVariantsResponse(),
        )
        self.store.save(record, user_id=current_user.user_id if current_user is not None else None)
        return record

    def create_product_from_pipeline(
        self,
        *,
        product: ProductPipelineResponse,
        current_user: AuthenticatedUser | None = None,
        status: str = "draft",
    ) -> ProductRecordResponse:
        record = ProductRecordResponse(
            id=self._new_id(),
            status=status,  # type: ignore[arg-type]
            created_at=self._timestamp(),
            updated_at=self._timestamp(),
            run_id=self.output_service.create_run_dir().name,
            product=product,
            variants=MarketplaceVariantsResponse(),
        )
        self.store.save(record, user_id=current_user.user_id if current_user is not None else None)
        return record

    def create_imported_draft(
        self,
        *,
        title: str,
        sku: str,
        brand: str,
        category: str,
        product_type: str,
        description: str,
        price: str,
        quantity: str,
        color: str,
        size: str,
        material: str,
        image_url: str,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        product = self.build_imported_product(
            title=title,
            sku=sku,
            brand=brand,
            category=category,
            product_type=product_type,
            description=description,
            price=price,
            quantity=quantity,
            color=color,
            size=size,
            material=material,
            image_url=image_url,
        )
        run_id = self.output_service.create_run_dir().name
        record = ProductRecordResponse(
            id=self._new_id(),
            status="draft",
            created_at=self._timestamp(),
            updated_at=self._timestamp(),
            run_id=run_id,
            product=product,
            variants=MarketplaceVariantsResponse(),
        )
        self.store.save(record, user_id=current_user.user_id if current_user is not None else None)
        return record

    def build_imported_product(
        self,
        *,
        title: str,
        sku: str,
        brand: str,
        category: str,
        product_type: str,
        description: str,
        price: str,
        quantity: str,
        color: str,
        size: str,
        material: str,
        image_url: str,
    ) -> ProductPipelineResponse:
        normalized_title = title or "Untitled Imported Product"
        attributes = {
            key: value
            for key, value in {
                "sku": sku,
                "brand": brand,
                "price": price,
                "quantity": quantity,
                "color": color,
                "size": size,
                "material": material,
            }.items()
            if value.strip()
        }
        core = CoreProductResponse(
            normalized_title=normalized_title,
            category=category or "General Merchandise",
            product_type=product_type or "general product",
            product_summary=description or "",
            features=[],
            attributes=attributes,
            source_title=title or normalized_title,
            vision_confidence=0.0,
        )
        amazon = AmazonResponse(
            title=normalized_title,
            bullet_points=[],
            description=description or "",
            backend_search_terms=[],
            structured_attributes={key.title(): value for key, value in attributes.items()},
        )
        ebay = EbayResponse(
            title=normalized_title[:80],
            item_specifics={key.title(): value for key, value in attributes.items()},
            condition="New",
            listing_notes=description or "",
        )
        etsy = EtsyResponse(
            title=normalized_title[:140],
            description=description or "",
            tags=[],
            materials=[material] if material else [],
            occasion="",
            seo_keywords=[],
        )
        tiktok = TikTokResponse(
            title=normalized_title,
            social_description=description or "",
            hashtags=[],
        )
        shopify = ShopifyResponse(
            title=normalized_title,
            body_html=f"<p>{description}</p>" if description else "",
            tags=[],
            product_type=product_type or "",
            seo_title=normalized_title[:70],
            seo_description=(description or "")[:180],
            category=category or "",
            metafields={
                key: value
                for key, value in {
                    "color": color.title() if color else "",
                    "footwear_material": material.title() if material else "",
                }.items()
                if value
            },
        )
        return ProductPipelineResponse(
            core=core,
            amazon=amazon,
            ebay=ebay,
            etsy=etsy,
            tiktok=tiktok,
            shopify=shopify,
            images=self._empty_generated_images(image_url=image_url),
            intelligence=self._empty_intelligence(),
        )

    def list_products(self, current_user: AuthenticatedUser | None = None) -> list[ProductListItemResponse]:
        return self.store.list(user_id=current_user.user_id if current_user is not None else None)

    def list_products_paginated(
        self,
        *,
        page: int,
        page_size: int,
        current_user: AuthenticatedUser | None = None,
    ) -> PaginatedProductListResponse:
        return self.store.list_paginated(
            page=page,
            page_size=page_size,
            user_id=current_user.user_id if current_user is not None else None,
        )

    def get_product(self, product_id: str, current_user: AuthenticatedUser | None = None) -> ProductRecordResponse:
        record = self.store.get(product_id, user_id=current_user.user_id if current_user is not None else None)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found.")
        return record

    def update_product(
        self,
        product_id: str,
        payload: ProductUpdateRequest,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        record = self.get_product(product_id, current_user=current_user)
        product = record.product

        if payload.core is not None:
            product = product.model_copy(
                update={
                    "core": product.core.model_copy(update=payload.core.model_dump(exclude_unset=True))
                }
            )
        if payload.amazon is not None:
            product = product.model_copy(
                update={
                    "amazon": product.amazon.model_copy(update=payload.amazon.model_dump(exclude_unset=True))
                }
            )
        if payload.tiktok is not None:
            product = product.model_copy(
                update={
                    "tiktok": product.tiktok.model_copy(update=payload.tiktok.model_dump(exclude_unset=True))
                }
            )
        if payload.ebay is not None:
            product = product.model_copy(
                update={
                    "ebay": product.ebay.model_copy(update=payload.ebay.model_dump(exclude_unset=True))
                }
            )
        if payload.etsy is not None:
            product = product.model_copy(
                update={
                    "etsy": product.etsy.model_copy(update=payload.etsy.model_dump(exclude_unset=True))
                }
            )
        if payload.shopify is not None:
            product = product.model_copy(
                update={
                    "shopify": product.shopify.model_copy(update=payload.shopify.model_dump(exclude_unset=True))
                }
            )

        updated = record.model_copy(
            update={
                "product": product,
                "updated_at": self._timestamp(),
            }
        )
        self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
        return updated

    async def upload_source_image(
        self,
        product_id: str,
        image: ImagePayload,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        record = self.get_product(product_id, current_user=current_user)
        product_dir = self.store.get_product_dir(record.id)
        source_asset = self.image_agent._save_source(image=image, run_dir=product_dir, output_service=self.output_service)
        cutout_asset = await self.image_agent._build_cutout(
            image=image,
            core_data=record.product.core,
            run_dir=product_dir,
            output_service=self.output_service,
        )
        images = self._pending_generated_images(
            source_asset=source_asset,
            transparent_cutout=cutout_asset,
        )
        validation = self.pipeline.validation.validate_pipeline(
            core=record.product.core,
            amazon=record.product.amazon,
            ebay=record.product.ebay,
            etsy=record.product.etsy,
            tiktok=record.product.tiktok,
            shopify=record.product.shopify,
            images=images,
        )
        updated_product = record.product.model_copy(
            update={
                "images": images,
                "intelligence": record.product.intelligence.model_copy(update={"validation": validation}),
            }
        )
        updated = record.model_copy(update={"product": updated_product, "updated_at": self._timestamp()})
        self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
        return updated

    def delete_product(
        self,
        product_id: str,
        current_user: AuthenticatedUser | None = None,
    ) -> None:
        record = self.get_product(product_id, current_user=current_user)
        user_id = current_user.user_id if current_user is not None else None
        deleted = False
        if hasattr(self.store, "delete"):
            deleted = bool(self.store.delete(product_id, user_id=user_id))  # type: ignore[attr-defined]
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found.")
        del record

    async def optimize_product(
        self,
        product_id: str,
        payload: ProductOptimizationRequest,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        record = self.get_product(product_id, current_user=current_user)
        current_core = record.product.core
        research = await self.pipeline.research.build_research_bundle(current_core)
        current_core = await self._seed_default_core_price(current_core, research)
        current_product = record.product.model_copy(update={"core": current_core})
        seo = await self.pipeline.seo.process(current_core, research)

        optimized = await self.optimization_agent.process(
            product=current_product,
            research=research,
            seo=seo,
            marketplaces=payload.marketplaces,
            optimize_core=payload.optimize_core,
        )
        optimized_core = self.optimization_agent.coerce_core(optimized.get("core"), current_product.core)
        optimized_amazon = self.optimization_agent.coerce_amazon(optimized.get("amazon"), current_product.amazon)
        optimized_ebay = self.optimization_agent.coerce_ebay(optimized.get("ebay"), current_product.ebay)
        optimized_etsy = self.optimization_agent.coerce_etsy(optimized.get("etsy"), current_product.etsy)
        optimized_tiktok = self.optimization_agent.coerce_tiktok(optimized.get("tiktok"), current_product.tiktok)
        optimized_shopify = self.optimization_agent.coerce_shopify(optimized.get("shopify"), current_product.shopify)

        validation = self.pipeline.validation.validate_pipeline(
            core=optimized_core,
            amazon=optimized_amazon,
            ebay=optimized_ebay,
            etsy=optimized_etsy,
            tiktok=optimized_tiktok,
            shopify=optimized_shopify,
            images=current_product.images,
        )
        optimized_product = ProductPipelineResponse(
            core=optimized_core,
            amazon=optimized_amazon,
            ebay=optimized_ebay,
            etsy=optimized_etsy,
            tiktok=optimized_tiktok,
            shopify=optimized_shopify,
            images=current_product.images,
            intelligence={
                "research": research,
                "seo": seo,
                "validation": validation,
            },
        )
        updated = record.model_copy(
            update={
                "product": optimized_product,
                "updated_at": self._timestamp(),
            }
        )
        self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
        return updated

    async def optimize_marketplace(
        self,
        product_id: str,
        marketplace: MarketplaceLiteral,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        payload = ProductOptimizationRequest(marketplaces=[marketplace], optimize_core=False)
        return await self.optimize_product(product_id, payload, current_user=current_user)

    async def regenerate_marketplace(
        self,
        product_id: str,
        marketplace: MarketplaceLiteral,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        return await self.generate_product_images(
            product_id,
            marketplaces=[marketplace],
            current_user=current_user,
        )

    async def generate_product_images(
        self,
        product_id: str,
        *,
        marketplaces: list[MarketplaceLiteral],
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        record = self.get_product(product_id, current_user=current_user)
        source_image = self._load_source_image(record)
        core = record.product.core
        research = await self.pipeline.research.build_research_bundle(core)
        core = await self._seed_default_core_price(core, research)
        seo = await self.pipeline.seo.process(core, research)
        amazon = record.product.amazon
        ebay = record.product.ebay
        etsy = record.product.etsy
        tiktok = record.product.tiktok
        shopify = record.product.shopify
        text_content = await self._generate_marketplace_content(core=core, research=research, seo=seo)
        if "amazon" in marketplaces:
            amazon = text_content["amazon"]
        if "ebay" in marketplaces:
            ebay = text_content["ebay"]
        if "etsy" in marketplaces:
            etsy = text_content["etsy"]
        if "tiktok" in marketplaces:
            tiktok = text_content["tiktok"]
        if "shopify" in marketplaces:
            shopify = text_content["shopify"]

        product_dir = self.store.get_product_dir(record.id)
        generated_images = await self._generate_marketplace_images(
            source=source_image,
            existing_images=record.product.images,
            core_data=core,
            amazon_data=amazon,
            ebay_data=ebay,
            etsy_data=etsy,
            tiktok_data=tiktok,
            shopify_data=shopify,
            marketplaces=marketplaces,
            run_dir=product_dir,
        )
        latest_record = self.get_product(product_id, current_user=current_user)
        latest_product = latest_record.product
        images = latest_product.images.model_copy(
            update={marketplace: getattr(generated_images, marketplace) for marketplace in marketplaces}
        )
        merged_amazon = amazon if "amazon" in marketplaces else latest_product.amazon
        merged_ebay = ebay if "ebay" in marketplaces else latest_product.ebay
        merged_etsy = etsy if "etsy" in marketplaces else latest_product.etsy
        merged_tiktok = tiktok if "tiktok" in marketplaces else latest_product.tiktok
        merged_shopify = shopify if "shopify" in marketplaces else latest_product.shopify
        validation = self.pipeline.validation.validate_pipeline(
            core=latest_product.core,
            amazon=merged_amazon,
            ebay=merged_ebay,
            etsy=merged_etsy,
            tiktok=merged_tiktok,
            shopify=merged_shopify,
            images=images,
        )
        product = ProductPipelineResponse(
            core=core,
            amazon=merged_amazon,
            ebay=merged_ebay,
            etsy=merged_etsy,
            tiktok=merged_tiktok,
            shopify=merged_shopify,
            images=images,
            intelligence={
                "research": research,
                "seo": seo,
                "validation": validation,
            },
        )
        updated = latest_record.model_copy(update={"product": product, "updated_at": self._timestamp()})
        self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
        return updated

    async def _build_generated_product(
        self,
        *,
        image: ImagePayload,
        title: str,
        image_marketplaces: list[MarketplaceLiteral],
    ) -> dict[str, object]:
        run_dir = self.output_service.create_run_dir()
        vision_data = await self.pipeline.vision.process(image)
        self.output_service.save_json(run_dir, "vision", vision_data.model_dump())
        core_data = await self.pipeline.core.process(title, vision_data)
        core_data = await self.pipeline.attribute_mapper.process(core_data, vision_data)
        research = await self.pipeline.research.build_research_bundle(core_data)
        core_data = await self._seed_default_core_price(core_data, research)
        self.output_service.save_json(run_dir, "core", core_data.model_dump())
        self.output_service.save_json(run_dir, "research", research.model_dump())
        seo = await self.pipeline.seo.process(core_data, research)
        self.output_service.save_json(run_dir, "seo", seo.model_dump())

        text_content = await self._generate_marketplace_content(core=core_data, research=research, seo=seo)
        for marketplace, payload in text_content.items():
            self.output_service.save_json(run_dir, marketplace, payload.model_dump())

        source_asset = self.image_agent._save_source(image=image, run_dir=run_dir, output_service=self.output_service)
        cutout_asset = await self.image_agent._build_cutout(
            image=image,
            core_data=core_data,
            run_dir=run_dir,
            output_service=self.output_service,
        )
        images = self._pending_generated_images(
            source_asset=source_asset,
            transparent_cutout=cutout_asset,
        )
        if image_marketplaces:
            images = await self._generate_marketplace_images(
                source=image,
                existing_images=images,
                core_data=core_data,
                amazon_data=text_content["amazon"],
                ebay_data=text_content["ebay"],
                etsy_data=text_content["etsy"],
                tiktok_data=text_content["tiktok"],
                shopify_data=text_content["shopify"],
                marketplaces=image_marketplaces,
                run_dir=run_dir,
            )
        self.output_service.save_json(run_dir, "images", images.model_dump())
        validation = self.pipeline.validation.validate_pipeline(
            core=core_data,
            amazon=text_content["amazon"],
            ebay=text_content["ebay"],
            etsy=text_content["etsy"],
            tiktok=text_content["tiktok"],
            shopify=text_content["shopify"],
            images=images,
        )
        product = ProductPipelineResponse(
            core=core_data,
            amazon=text_content["amazon"],
            ebay=text_content["ebay"],
            etsy=text_content["etsy"],
            tiktok=text_content["tiktok"],
            shopify=text_content["shopify"],
            images=images,
            intelligence={
                "research": research,
                "seo": seo,
                "validation": validation,
            },
        )
        self.output_service.save_json(run_dir, "validation", validation.model_dump())
        self.output_service.save_json(run_dir, "final", product.model_dump())
        return {
            "run_id": run_dir.name,
            "product": product,
        }

    async def _generate_marketplace_content(
        self,
        *,
        core: CoreProductResponse,
        research: MarketResearchBundleResponse,
        seo: SeoInsightsResponse,
    ) -> dict[str, AmazonResponse | EbayResponse | EtsyResponse | TikTokResponse | ShopifyResponse]:
        amazon, ebay, etsy, tiktok, shopify = await asyncio.gather(
            self.amazon_agent.process(core, research=research.amazon, seo=seo),
            self.ebay_agent.process(core, research=research.ebay, seo=seo),
            self.etsy_agent.process(core, research=research.etsy, seo=seo),
            self.tiktok_agent.process(core, research=research.tiktok, seo=seo),
            self.shopify_agent.process(core, research=research.shopify, seo=seo),
        )
        return {
            "amazon": amazon,
            "ebay": ebay,
            "etsy": etsy,
            "tiktok": tiktok,
            "shopify": shopify,
        }

    async def _generate_marketplace_images(
        self,
        *,
        source: ImagePayload,
        existing_images: GeneratedImagesResponse,
        core_data: CoreProductResponse,
        amazon_data: AmazonResponse,
        ebay_data: EbayResponse,
        etsy_data: EtsyResponse,
        tiktok_data: TikTokResponse,
        shopify_data: ShopifyResponse,
        marketplaces: list[MarketplaceLiteral],
        run_dir: Path,
    ) -> GeneratedImagesResponse:
        async def generate_one(marketplace: MarketplaceLiteral) -> tuple[MarketplaceLiteral, ImageVariantResponse]:
            asset = await self.image_agent.regenerate_marketplace_asset(
                marketplace=marketplace,
                source=source,
                existing_images=existing_images,
                core_data=core_data,
                amazon_data=amazon_data,
                ebay_data=ebay_data,
                etsy_data=etsy_data,
                tiktok_data=tiktok_data,
                shopify_data=shopify_data,
                run_dir=run_dir,
                output_service=self.output_service,
            )
            return marketplace, asset

        unique_marketplaces = list(dict.fromkeys(marketplaces))
        generated = await asyncio.gather(*(generate_one(marketplace) for marketplace in unique_marketplaces))
        return existing_images.model_copy(update={marketplace: asset for marketplace, asset in generated})

    async def get_product_pricing_snapshot(
        self,
        product_id: str,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductPricingSnapshotResponse:
        record = self.get_product(product_id, current_user=current_user)
        core = record.product.core
        research = self._empty_pricing_research_bundle()
        core = await self._seed_default_core_price(core, research)
        if core != record.product.core:
            updated_product = record.product.model_copy(update={"core": core})
            record = record.model_copy(
                update={
                    "product": updated_product,
                    "updated_at": self._timestamp(),
                }
            )
            self.store.save(record, user_id=current_user.user_id if current_user is not None else None)
        markets = await asyncio.gather(
            self._build_marketplace_pricing_snapshot("amazon", research.amazon, core, listing_title=record.product.amazon.title),
            self._build_marketplace_pricing_snapshot("ebay", research.ebay, core, listing_title=record.product.ebay.title),
            self._build_marketplace_pricing_snapshot("etsy", research.etsy, core, listing_title=record.product.etsy.title),
            self._build_marketplace_pricing_snapshot("tiktok", research.tiktok, core, listing_title=record.product.tiktok.title),
            self._build_marketplace_pricing_snapshot("shopify", research.shopify, core, listing_title=record.product.shopify.title),
        )
        timestamp = self._timestamp()
        return ProductPricingSnapshotResponse(
            product_id=record.id,
            generated_at=timestamp,
            markets=markets,
        )

    async def query_product_pricing(
        self,
        product_name: str,
        marketplace: MarketplaceLiteral | None = None,
    ) -> DynamicPricingQueryResponse:
        core = self._build_query_core(product_name)
        research = self._empty_pricing_research_bundle()
        markets = [marketplace] if marketplace is not None else ["amazon", "ebay", "etsy", "tiktok", "shopify"]
        results = await asyncio.gather(
            *(self._build_marketplace_pricing_snapshot(market, self._research_for_marketplace(research, market), core, listing_title=product_name) for market in markets)
        )
        return DynamicPricingQueryResponse(
            query=product_name,
            generated_at=self._timestamp(),
            markets=list(results),
        )

    async def _build_marketplace_pricing_snapshot(
        self,
        marketplace: MarketplaceLiteral,
        research: MarketplaceResearchResponse,
        core: CoreProductResponse,
        listing_title: str = "",
    ) -> MarketplacePricingSnapshotResponse:
        if self._gemini_search_enabled():
            gemini_estimate = await self._build_gemini_pricing_estimate(marketplace, research, core, listing_title=listing_title)
            if gemini_estimate is not None:
                suggested = gemini_estimate["suggested_price_range"]
                recommended = gemini_estimate["recommended_price"]
                return MarketplacePricingSnapshotResponse(
                    marketplace=marketplace,
                    source_mode="gemini_search",
                    search_queries=gemini_estimate["search_queries"],
                    comparable_count=gemini_estimate["comparable_count"],
                    recommended_price=recommended,
                    currency=suggested.currency if suggested is not None else "USD",
                    suggested_price_range=suggested,
                    market_signal=gemini_estimate["market_signal"],
                    analysis_summary=gemini_estimate["analysis_summary"],
                    similar_listings=gemini_estimate.get("similar_listings") or [],
                )
            return MarketplacePricingSnapshotResponse(
                marketplace=marketplace,
                source_mode="insufficient_data",
                search_queries=self._build_gemini_search_queries(core, marketplace, research, listing_title=listing_title),
                comparable_count=0,
                recommended_price=None,
                currency="USD",
                suggested_price_range=None,
                market_signal=self._insufficient_pricing_signal(marketplace, research),
                analysis_summary=(
                    f"{self._product_identity_label(core)} on {marketplace.title()} "
                    "does not have enough reliable Google search pricing data yet."
                ),
                similar_listings=[],
            )

        return MarketplacePricingSnapshotResponse(
            marketplace=marketplace,
            source_mode="insufficient_data",
            search_queries=research.search_queries,
            comparable_count=0,
            recommended_price=None,
            currency="USD",
            suggested_price_range=None,
            market_signal=self._insufficient_pricing_signal(marketplace, research),
            analysis_summary=(
                f"{self._product_identity_label(core)} on {marketplace.title()} "
                "does not have enough reliable pricing data yet."
            ),
            similar_listings=[],
        )

    def add_size_variant(
        self,
        product_id: str,
        marketplace: MarketplaceLiteral,
        payload: VariantCreateRequest,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        record = self.get_product(product_id, current_user=current_user)
        variant = ProductVariantResponse(
            id=self._new_id(),
            marketplace=marketplace,
            variant_type="size",
            name=payload.name,
            value=payload.value or payload.name,
            image=None,
            created_at=self._timestamp(),
        )
        updated = self._append_variant(record, marketplace, variant)
        self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
        return updated

    def add_color_variant(
        self,
        product_id: str,
        marketplace: MarketplaceLiteral,
        payload: VariantCreateRequest,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        record = self.get_product(product_id, current_user=current_user)
        source_image = self._load_source_image(record)
        product_dir = self.store.get_product_dir(record.id)
        image_asset = self.image_agent.build_color_variant_asset(
            marketplace=marketplace,
            source=source_image,
            existing_images=record.product.images,
            color_name=payload.value or payload.name,
            title=record.product.core.normalized_title,
            run_dir=product_dir,
            output_service=self.output_service,
        )
        variant = ProductVariantResponse(
            id=self._new_id(),
            marketplace=marketplace,
            variant_type="color",
            name=payload.name,
            value=payload.value or payload.name,
            image=image_asset,
            created_at=self._timestamp(),
        )
        updated = self._append_variant(record, marketplace, variant)
        self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
        return updated

    def _append_variant(
        self,
        record: ProductRecordResponse,
        marketplace: MarketplaceLiteral,
        variant: ProductVariantResponse,
    ) -> ProductRecordResponse:
        variant_map = {
            "amazon": list(record.variants.amazon),
            "ebay": list(record.variants.ebay),
            "etsy": list(record.variants.etsy),
            "tiktok": list(record.variants.tiktok),
            "shopify": list(record.variants.shopify),
        }
        variant_map[marketplace].append(variant)
        return record.model_copy(
            update={
                "variants": MarketplaceVariantsResponse(**variant_map),
                "updated_at": self._timestamp(),
            }
        )

    @staticmethod
    def _has_reliable_market_pricing(research: MarketplaceResearchResponse, core: CoreProductResponse | None = None) -> bool:
        if research.source_mode != "live_api":
            return False
        listing_prices = ProductService._filtered_live_listing_prices(research, core)
        return len(listing_prices) >= 2

    @staticmethod
    def _best_live_pricing_reference(
        research: MarketResearchBundleResponse,
        core: CoreProductResponse,
    ) -> tuple[MarketplaceLiteral | None, MarketplaceResearchResponse | None]:
        candidates: list[tuple[MarketplaceLiteral, MarketplaceResearchResponse]] = [
            ("amazon", research.amazon),
            ("ebay", research.ebay),
            ("etsy", research.etsy),
            ("tiktok", research.tiktok),
            ("shopify", research.shopify),
        ]
        live_candidates = [
            (marketplace, market_research)
            for marketplace, market_research in candidates
            if ProductService._has_reliable_market_pricing(market_research, core)
        ]
        if not live_candidates:
            return None, None
        live_candidates.sort(
            key=lambda item: len(ProductService._filtered_live_listing_prices(item[1], core)),
            reverse=True,
        )
        return live_candidates[0]

    async def _build_gemini_pricing_estimate(
        self,
        marketplace: MarketplaceLiteral,
        research: MarketplaceResearchResponse,
        core: CoreProductResponse,
        listing_title: str = "",
    ) -> dict[str, Any] | None:
        base_payload = {
            "marketplace": marketplace,
            "product_identity": self._product_identity_label(core),
            "generated_listing_title": listing_title.strip(),
            "source_title": core.source_title,
            "category": core.category,
            "product_type": core.product_type,
            "product_summary": core.product_summary,
            "features": core.features,
            "attributes": core.attributes,
            "existing_search_queries": self._build_gemini_search_queries(core, marketplace, research, listing_title=listing_title),
            "response_contract": {
                "status": "ok or insufficient_data",
                "minimum": "number",
                "maximum": "number",
                "recommended": "number",
                "currency": "string",
                "market_signal": "string",
                "analysis_summary": "string",
                "search_queries": ["string"],
                "comparable_count": "integer",
            },
        }
        payload: dict[str, Any] | None = None
        for schema in (self._GEMINI_PRICING_SCHEMA, None):
            try:
                candidate = await self.pipeline.gemini_service.generate_structured_output(
                    system_prompt=PromptRegistry.get_gemini_pricing_search_prompt(),
                    user_payload=base_payload,
                    use_google_search=True,
                    schema=schema,
                )
            except GeminiServiceError:
                continue
            payload = self._normalize_gemini_pricing_payload(candidate)
            if payload is not None:
                break
        if payload is None:
            return None

        similar_listings = self._coerce_gemini_price_sources(payload.get("price_sources"))
        minimum = self._parse_price(payload.get("minimum"))
        maximum = self._parse_price(payload.get("maximum"))
        recommended = self._parse_price(payload.get("recommended"))
        comparable_count = int(self._parse_price(payload.get("comparable_count")) or 0)
        suggested: SuggestedPriceRangeResponse | None = None
        if minimum is not None and maximum is not None:
            if maximum < minimum:
                minimum, maximum = maximum, minimum
            if recommended is None:
                recommended = round((minimum + maximum) / 2, 2)
            else:
                recommended = min(max(recommended, minimum), maximum)
            suggested = SuggestedPriceRangeResponse(
                minimum=minimum,
                maximum=maximum,
                recommended=recommended,
                currency=str(payload.get("currency") or "USD").strip() or "USD",
                source="gemini_search",
            )
        else:
            suggested = self._pricing_range_from_sources(similar_listings, marketplace=marketplace)
            if suggested is not None:
                recommended = suggested.recommended
        if suggested is None:
            return None

        search_queries = payload.get("search_queries")
        if not isinstance(search_queries, list):
            search_queries = self._build_gemini_search_queries(core, marketplace, research, listing_title=listing_title)
        return {
            "suggested_price_range": suggested,
            "recommended_price": recommended,
            "market_signal": str(payload.get("market_signal") or "").strip() or f"{marketplace.title()} Gemini search pricing",
            "analysis_summary": str(payload.get("analysis_summary") or "").strip()
            or self._analysis_summary_for_marketplace(
                self._product_identity_label(core),
                marketplace,
                research,
                recommended,
                core.product_summary,
                pricing_mode="gemini_search",
                suggested_range=suggested,
            ),
            "search_queries": [str(item).strip() for item in search_queries if str(item).strip()],
            "comparable_count": max(0, comparable_count or len(similar_listings)),
            "similar_listings": similar_listings,
        }

    @staticmethod
    def _empty_pricing_research_bundle() -> MarketResearchBundleResponse:
        return MarketResearchBundleResponse(
            amazon=MarketplaceResearchResponse(marketplace="amazon"),
            ebay=MarketplaceResearchResponse(marketplace="ebay"),
            etsy=MarketplaceResearchResponse(marketplace="etsy"),
            tiktok=MarketplaceResearchResponse(marketplace="tiktok"),
            shopify=MarketplaceResearchResponse(marketplace="shopify"),
        )

    @staticmethod
    def _build_query_core(product_name: str) -> CoreProductResponse:
        normalized = product_name.strip()
        query_keywords = title_keywords(normalized)[:8]
        category = query_keywords[0].title() if query_keywords else normalized or "query"
        product_type = query_keywords[1].title() if len(query_keywords) > 1 else category
        attributes: dict[str, str] = {
            "query": normalized,
            "query_keywords": ", ".join(query_keywords),
        }

        return CoreProductResponse(
            normalized_title=normalized,
            category=category,
            product_type=product_type,
            product_summary=f"Dynamic pricing lookup for {normalized}.",
            features=[
                f"Query-driven pricing for {normalized}.",
                f"Search keywords: {', '.join(query_keywords) if query_keywords else normalized}.",
                "Uses live Google search evidence and ignores mismatched variants.",
            ],
            attributes=attributes,
            source_title=normalized,
            vision_confidence=0.0,
        )

    @staticmethod
    def _normalize_gemini_pricing_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None

        normalized = dict(payload)
        key_aliases = {
            "min": "minimum",
            "min_price": "minimum",
            "minimum_price": "minimum",
            "low": "minimum",
            "max": "maximum",
            "max_price": "maximum",
            "maximum_price": "maximum",
            "high": "maximum",
            "recommended_price": "recommended",
            "default_price": "recommended",
            "price": "recommended",
            "estimated_price": "recommended",
            "count": "comparable_count",
            "matches_found": "comparable_count",
            "query_candidates": "search_queries",
            "sources": "price_sources",
            "listings": "price_sources",
            "similar_listings": "price_sources",
        }
        for alias, canonical in key_aliases.items():
            if canonical not in normalized and alias in normalized:
                normalized[canonical] = normalized[alias]

        status = str(normalized.get("status") or "").strip().lower()
        price_sources = normalized.get("price_sources")
        has_price_sources = isinstance(price_sources, list) and any(
            isinstance(item, dict) and item.get("price") is not None for item in price_sources
        )
        if not status:
            if any(normalized.get(key) is not None for key in ("minimum", "maximum", "recommended")) or has_price_sources:
                normalized["status"] = "ok"
            else:
                return None
        elif status not in {"ok", "insufficient_data"}:
            normalized["status"] = "ok" if any(
                normalized.get(key) is not None for key in ("minimum", "maximum", "recommended")
            ) or has_price_sources else "insufficient_data"

        if str(normalized.get("status")).strip() != "ok" and not has_price_sources:
            return None
        return normalized

    @staticmethod
    def _coerce_gemini_price_sources(value: object) -> list[ResearchEvidenceResponse]:
        if not isinstance(value, list):
            return []

        sources: list[ResearchEvidenceResponse] = []
        for item in value[:6]:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or "").strip()
            title = str(item.get("title") or "").strip()
            if not source or not title:
                continue
            price_value = None
            for key in ("price", "sale_price", "list_price", "regular_price", "current_price", "observed_price"):
                price_value = ProductService._parse_price(item.get(key))
                if price_value is not None:
                    break
            sources.append(
                ResearchEvidenceResponse(
                    source=source,
                    title=title,
                    url=str(item.get("url") or "").strip() or None,
                    price=price_value,
                    currency=str(item.get("currency") or "USD").strip() or "USD",
                    relevance_score=0.92,
                    attributes={},
                    observations=[],
                )
            )
        return sources

    @staticmethod
    def _pricing_range_from_sources(
        sources: list[ResearchEvidenceResponse],
        *,
        marketplace: MarketplaceLiteral,
    ) -> SuggestedPriceRangeResponse | None:
        prices = sorted(
            float(source.price)
            for source in sources
            if source.price is not None and source.price > 0
        )
        if len(prices) < 2:
            return None

        observed_min = prices[0]
        observed_max = prices[-1]
        observed_median = median(prices)
        observed_mean = mean(prices)
        spread = max(observed_max - observed_min, max(0.08 * observed_median, 5.0))
        marketplace_bias = {
            "amazon": 1.0,
            "ebay": 0.99,
            "etsy": 1.02,
            "tiktok": 0.97,
            "shopify": 1.04,
        }[marketplace]
        recommended = round(max(0.01, ((observed_median + observed_mean) / 2) * marketplace_bias), 2)
        padding = max(spread * 0.18, recommended * 0.06)
        minimum = round(max(0.01, min(observed_min, recommended - padding)), 2)
        maximum = round(max(observed_max, recommended + padding), 2)
        return SuggestedPriceRangeResponse(
            minimum=minimum,
            maximum=maximum,
            recommended=recommended,
            currency=next((source.currency for source in sources if source.currency), "USD"),
            source="gemini_search_sources",
        )

    def _gemini_search_enabled(self) -> bool:
        return bool(getattr(self.pipeline.gemini_service, "enabled", False))

    @staticmethod
    def _insufficient_pricing_signal(marketplace: MarketplaceLiteral, research: MarketplaceResearchResponse) -> str:
        if research.search_queries:
            return f"{marketplace.title()} pricing needs stronger matches for: {research.search_queries[0]}"
        return f"{marketplace.title()} pricing data is insufficient right now."

    @staticmethod
    def _research_for_marketplace(research: MarketResearchBundleResponse, marketplace: MarketplaceLiteral) -> MarketplaceResearchResponse:
        if marketplace == "amazon":
            return research.amazon
        if marketplace == "ebay":
            return research.ebay
        if marketplace == "etsy":
            return research.etsy
        if marketplace == "tiktok":
            return research.tiktok
        return research.shopify

    async def _seed_default_core_price(
        self,
        core: CoreProductResponse,
        research: MarketResearchBundleResponse,
    ) -> CoreProductResponse:
        current_price = self._parse_price(core.attributes.get("price"))
        if current_price is not None:
            return core

        seeded_price = await self._estimate_initial_draft_price(core, research)
        if seeded_price is None:
            return core

        return core.model_copy(
            update={
                "attributes": {
                    **core.attributes,
                    "price": f"{seeded_price:.2f}",
                }
            }
        )

    async def _estimate_initial_draft_price(
        self,
        core: CoreProductResponse,
        research: MarketResearchBundleResponse,
    ) -> float | None:
        if self._gemini_search_enabled():
            gemini_estimate = await self._build_gemini_pricing_estimate("shopify", research.shopify, core)
            if gemini_estimate is not None:
                recommended = self._parse_price(gemini_estimate.get("recommended_price"))
                if recommended is not None:
                    return round(recommended, 2)
        return None

    @staticmethod
    def _default_price_from_band(
        suggested: SuggestedPriceRangeResponse | None,
        research: MarketplaceResearchResponse,
        core: CoreProductResponse,
        marketplace: MarketplaceLiteral,
    ) -> float | None:
        if suggested is None:
            return None

        minimum = suggested.minimum
        maximum = suggested.maximum
        recommended = suggested.recommended
        if maximum <= minimum:
            return round(recommended, 2)

        identity = " ".join(
            [
                core.normalized_title,
                core.source_title,
                core.category,
                core.product_type,
                core.product_summary,
                " ".join(core.features),
            ]
        ).lower()
        is_high_value_electronics = any(
            keyword in identity
            for keyword in (
                "playstation",
                "ps5",
                "xbox",
                "nintendo switch",
                "console",
                "gaming",
                "electronics",
            )
        )

        anchor_ratio = 0.78 if is_high_value_electronics else 0.66
        if research.source_mode == "live_api":
            anchor_ratio = max(anchor_ratio, 0.76)
        elif marketplace in {"amazon", "shopify"}:
            anchor_ratio = max(anchor_ratio, 0.7)

        strategic_price = minimum + ((maximum - minimum) * anchor_ratio)
        strategic_price = max(strategic_price, recommended)
        return round(min(maximum, strategic_price), 2)

    @staticmethod
    def _suggested_price_for_marketplace(
        core: CoreProductResponse,
        marketplace: MarketplaceLiteral,
        research: MarketplaceResearchResponse,
    ) -> SuggestedPriceRangeResponse | None:
        listing_prices = ProductService._filtered_live_listing_prices(research, core)
        if not listing_prices:
            listing_prices = [listing.price for listing in research.similar_listings if listing.price is not None]
        if listing_prices:
            observed_prices = sorted(listing_prices)
            observed_min = observed_prices[0]
            observed_max = observed_prices[-1]
            observed_median = median(observed_prices)
            observed_mean = mean(observed_prices)
            spread = observed_max - observed_min
            if spread <= 0:
                spread = max(0.08 * observed_median, 5.0)

            marketplace_bias = {
                "amazon": 1.0,
                "ebay": 0.99,
                "etsy": 1.02,
                "tiktok": 0.97,
                "shopify": 1.04,
            }[marketplace]
            recommended = round(max(0.01, (observed_median + observed_mean) / 2 * marketplace_bias), 2)

            padding = max(spread * 0.18, recommended * 0.06)
            minimum = round(max(0.01, min(observed_min, recommended - padding)), 2)
            maximum = round(max(observed_max, recommended + padding), 2)
        else:
            return None

        return SuggestedPriceRangeResponse(
            minimum=round(minimum, 2),
            maximum=round(maximum, 2),
            recommended=recommended,
            currency="USD",
            source=research.source_mode or "market_research",
        )

    @staticmethod
    def _filtered_live_listing_prices(
        research: MarketplaceResearchResponse,
        core: CoreProductResponse | None,
    ) -> list[float]:
        raw_prices = [listing.price for listing in research.similar_listings if listing.price is not None and listing.price > 0]
        if research.source_mode != "live_api" or core is None:
            return raw_prices

        matched_prices = [
            float(listing.price)
            for listing in research.similar_listings
            if listing.price is not None
            and listing.price > 0
            and ProductService._listing_matches_product_identity(core, listing.title)
        ]
        if len(matched_prices) < 2:
            matched_prices = raw_prices
        if not matched_prices:
            return []

        matched_prices.sort()
        center = median(matched_prices)
        tolerance = max(10.0, center * 0.03)
        clustered = [price for price in matched_prices if abs(price - center) <= tolerance]
        if len(clustered) >= 2:
            return clustered
        return matched_prices

    @staticmethod
    def _listing_matches_product_identity(core: CoreProductResponse, listing_title: str) -> bool:
        product_tokens = set(title_keywords(f"{core.normalized_title} {core.source_title}"))
        listing_tokens = set(title_keywords(listing_title))
        if not product_tokens or not listing_tokens:
            return False

        important_tokens = ProductService._important_identity_tokens(core)
        matched_important = {token for token in important_tokens if token in listing_tokens}
        required_numeric = {token for token in important_tokens if any(char.isdigit() for char in token)}
        if required_numeric and not required_numeric.issubset(listing_tokens):
            return False

        overlap = len(product_tokens & listing_tokens)
        overlap_ratio = overlap / max(1, min(len(product_tokens), len(listing_tokens)))
        if important_tokens and len(matched_important) < max(1, min(2, len(important_tokens))):
            return False
        return overlap_ratio >= 0.45 or overlap >= 3

    @staticmethod
    def _important_identity_tokens(core: CoreProductResponse) -> set[str]:
        raw_tokens = re.findall(r"[a-zA-Z0-9]+", f"{core.normalized_title} {core.source_title}".lower())
        important: set[str] = set()
        special_tokens = {"rtx", "gtx", "rx", "ps5", "xbox", "oled", "ti", "super", "pro", "max"}
        for token in raw_tokens:
            if len(token) >= 3 and any(char.isdigit() for char in token):
                important.add(token)
            elif token in special_tokens:
                important.add(token)
        brand = str(core.attributes.get("brand") or "").strip().lower()
        model = str(core.attributes.get("model") or core.attributes.get("model_number") or "").strip().lower()
        if brand:
            important.add(brand)
        if model:
            important.update(part for part in re.findall(r"[a-zA-Z0-9]+", model) if part)
        return important

    @staticmethod
    def _build_gemini_search_queries(
        core: CoreProductResponse,
        marketplace: MarketplaceLiteral,
        research: MarketplaceResearchResponse,
        listing_title: str = "",
    ) -> list[str]:
        brand = str(core.attributes.get("brand") or "").strip()
        model = str(core.attributes.get("model") or core.attributes.get("model_number") or "").strip()
        color = str(core.attributes.get("color") or "").strip()
        material = str(core.attributes.get("material") or "").strip()
        style = str(core.attributes.get("style") or "").strip()
        query_keywords = str(core.attributes.get("query_keywords") or "").strip()
        identity = ProductService._product_identity_label(core)
        title = core.source_title.strip() or identity
        current_listing_title = listing_title.strip()
        important_tokens = sorted(ProductService._important_identity_tokens(core))
        compact_identity = " ".join(token.upper() if token in {"rtx", "gtx", "rx"} else token for token in important_tokens[:6]).strip()
        visual_identity = " ".join(part for part in [brand, model, color, material, style, core.product_type] if part).strip()
        category_identity = " ".join(part for part in [brand, core.category, core.product_type, color, material] if part).strip()
        query_identity = " ".join(part for part in [current_listing_title, query_keywords, identity, title] if part).strip()

        queries = unique_strings(
            [
                f"{query_identity} price".strip(),
                f"{query_identity} buy".strip(),
                f"{current_listing_title} price".strip(),
                f"{identity} price",
                f"{title} price",
                f"{current_listing_title} buy".strip(),
                f"{identity} buy".strip(),
                f"{identity} retail price".strip(),
                f"{identity} store price".strip(),
                f"{identity} Google Shopping".strip(),
                f"{identity} MSRP".strip(),
                f"{visual_identity} price".strip(),
                f"{category_identity} price".strip(),
                f"{brand} {model} price".strip(),
                f"{brand} {compact_identity} price".strip(),
                f"{identity} {marketplace} price".strip(),
                f"{marketplace} {identity} price".strip(),
                f"{identity} current price".strip(),
            ]
            + list(research.search_queries),
            limit=12,
        )
        return [query for query in queries if query.strip()]

    @staticmethod
    def _market_signal_for_marketplace(marketplace: MarketplaceLiteral, research: MarketplaceResearchResponse) -> str:
        signals = ", ".join(research.keyword_signals[:4])
        source = research.source_mode.replace("_", " ")
        if signals:
            return f"{marketplace.title()} {source}: {signals}"
        return f"{marketplace.title()} {source}"

    @staticmethod
    def _analysis_summary_for_marketplace(
        product_label: str,
        marketplace: MarketplaceLiteral,
        research: MarketplaceResearchResponse,
        default_price: float | None,
        product_summary: str = "",
        pricing_mode: str = "live_api",
        suggested_range: SuggestedPriceRangeResponse | None = None,
    ) -> str:
        if default_price is None:
            return (
                f"{product_label} on {marketplace.title()} does not have enough reliable pricing evidence yet."
            )
        if pricing_mode == "live_api":
            source_label = "live market data"
        elif pricing_mode == "cross_market_live_reference":
            source_label = "cross-market live pricing reference"
        elif pricing_mode == "gemini_search":
            source_label = "Gemini search-grounded pricing"
        else:
            source_label = "pricing analysis"
        summary_min = suggested_range.minimum if suggested_range is not None else research.price_min
        summary_max = suggested_range.maximum if suggested_range is not None else research.price_max
        if summary_min is not None and summary_max is not None:
            summary_part = f" {product_summary.strip()}" if product_summary.strip() else ""
            return (
                f"{product_label} on {marketplace.title()} supports a default price around ${default_price:.2f}."
                f"{summary_part}"
                f" {source_label} bands from ${summary_min:.2f} to ${summary_max:.2f}."
            )
        summary_part = f" {product_summary.strip()}" if product_summary.strip() else ""
        return f"{product_label} on {marketplace.title()} {source_label} recommends a default price around ${default_price:.2f}.{summary_part}"

    @staticmethod
    def _product_identity_label(core: CoreProductResponse) -> str:
        label = core.normalized_title.strip() or core.source_title.strip() or "Product"
        return re.sub(r"\s+", " ", label).strip()

    @staticmethod
    def _parse_price(value: object) -> float | None:
        try:
            parsed = float(str(value).strip())
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        return round(parsed, 2)


    @staticmethod
    def _new_id() -> str:
        return uuid4().hex

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _empty_generated_images(*, image_url: str) -> GeneratedImagesResponse:
        source_path = image_url.strip()
        return GeneratedImagesResponse(
            source=ProductService._placeholder_image(
                marketplace="source",
                generation_mode="source_passthrough" if source_path else "missing_source",
                prompt="Imported source image reference." if source_path else "No source image was provided during import.",
                absolute_path=source_path,
                expected_background="source",
                mime_type="image/jpeg",
                has_alpha=False,
            ),
            transparent_cutout=ProductService._placeholder_image(
                marketplace="transparent_cutout",
                generation_mode="pending_import_source",
                prompt="Transparent cutout will be generated after source imagery is available.",
                absolute_path="",
                expected_background="transparent",
                mime_type="image/png",
                has_alpha=True,
            ),
            amazon=ProductService._placeholder_image(
                marketplace="amazon",
                generation_mode="pending_import_source",
                prompt="Amazon image not generated yet.",
                absolute_path="",
                expected_background="white",
                mime_type="image/png",
                has_alpha=False,
            ),
            ebay=ProductService._placeholder_image(
                marketplace="ebay",
                generation_mode="pending_import_source",
                prompt="eBay image not generated yet.",
                absolute_path="",
                expected_background="white",
                mime_type="image/png",
                has_alpha=False,
            ),
            etsy=ProductService._placeholder_image(
                marketplace="etsy",
                generation_mode="pending_import_source",
                prompt="Etsy image not generated yet.",
                absolute_path="",
                expected_background="styled",
                mime_type="image/png",
                has_alpha=False,
            ),
            tiktok=ProductService._placeholder_image(
                marketplace="tiktok",
                generation_mode="pending_import_source",
                prompt="TikTok image not generated yet.",
                absolute_path="",
                expected_background="styled",
                mime_type="image/png",
                has_alpha=False,
            ),
            shopify=ProductService._placeholder_image(
                marketplace="shopify",
                generation_mode="pending_import_source",
                prompt="Shopify image not generated yet.",
                absolute_path=source_path,
                expected_background="styled",
                mime_type="image/png",
                has_alpha=False,
            ),
        )

    @staticmethod
    def _pending_generated_images(
        *,
        source_asset: ImageVariantResponse,
        transparent_cutout: ImageVariantResponse | None,
    ) -> GeneratedImagesResponse:
        return GeneratedImagesResponse(
            source=source_asset,
            transparent_cutout=transparent_cutout
            or ProductService._placeholder_image(
                marketplace="transparent_cutout",
                generation_mode="pending_cutout",
                prompt="Transparent cutout is not available yet.",
                absolute_path="",
                expected_background="transparent",
                mime_type="image/png",
                has_alpha=True,
            ),
            amazon=ProductService._placeholder_image(
                marketplace="amazon",
                generation_mode="pending_marketplace_generation",
                prompt="Generate Amazon image from the transparent cutout when needed.",
                absolute_path="",
                expected_background="white",
                mime_type="image/png",
                has_alpha=False,
            ),
            ebay=ProductService._placeholder_image(
                marketplace="ebay",
                generation_mode="pending_marketplace_generation",
                prompt="Generate eBay image from the transparent cutout when needed.",
                absolute_path="",
                expected_background="white",
                mime_type="image/png",
                has_alpha=False,
            ),
            etsy=ProductService._placeholder_image(
                marketplace="etsy",
                generation_mode="pending_marketplace_generation",
                prompt="Generate Etsy image from the transparent cutout when needed.",
                absolute_path="",
                expected_background="opaque",
                mime_type="image/png",
                has_alpha=False,
            ),
            tiktok=ProductService._placeholder_image(
                marketplace="tiktok",
                generation_mode="pending_marketplace_generation",
                prompt="Generate TikTok Shop image from the transparent cutout when needed.",
                absolute_path="",
                expected_background="opaque",
                mime_type="image/png",
                has_alpha=False,
            ),
            shopify=ProductService._placeholder_image(
                marketplace="shopify",
                generation_mode="pending_marketplace_generation",
                prompt="Generate Shopify image from the transparent cutout when needed.",
                absolute_path="",
                expected_background="opaque",
                mime_type="image/png",
                has_alpha=False,
            ),
        )

    @staticmethod
    def _placeholder_image(
        *,
        marketplace: str,
        generation_mode: str,
        prompt: str,
        absolute_path: str,
        expected_background: str,
        mime_type: str,
        has_alpha: bool,
    ) -> ImageVariantResponse:
        return ImageVariantResponse(
            marketplace=marketplace,
            relative_path="",
            absolute_path=absolute_path,
            prompt=prompt,
            generation_mode=generation_mode,
            mime_type=mime_type,
            validation=ImageValidationResponse(
                passed=bool(absolute_path) if marketplace == "source" else False,
                width=None,
                height=None,
                format=None,
                has_alpha=has_alpha,
                file_size_bytes=0,
                expected_width=None,
                expected_height=None,
                expected_background=expected_background,
                errors=[] if absolute_path else ["No image generated yet."],
                mime_type=mime_type,
            ),
        )

    @staticmethod
    def _empty_intelligence() -> IntelligenceLayerResponse:
        def empty_market_research(marketplace: str) -> MarketplaceResearchResponse:
            return MarketplaceResearchResponse(marketplace=marketplace)

        empty_section = SectionValidationResponse(passed=True, issues=[])
        return IntelligenceLayerResponse(
            research=MarketResearchBundleResponse(
                amazon=empty_market_research("amazon"),
                ebay=empty_market_research("ebay"),
                etsy=empty_market_research("etsy"),
                tiktok=empty_market_research("tiktok"),
                shopify=empty_market_research("shopify"),
            ),
            seo=SeoInsightsResponse(),
            validation=PipelineValidationResponse(
                core=empty_section,
                amazon=empty_section,
                ebay=empty_section,
                etsy=empty_section,
                tiktok=empty_section,
                shopify=empty_section,
                images=empty_section,
            ),
        )

    @staticmethod
    def _load_source_image(record: ProductRecordResponse) -> ImagePayload:
        source = record.product.images.source
        if source.absolute_path.startswith("http://") or source.absolute_path.startswith("https://"):
            remote_url = ProductService._safe_remote_url(source.absolute_path)
            with urlopen(remote_url, timeout=30) as response:
                payload = response.read()
            filename = Path(source.relative_path or Path(urlsplit(remote_url).path).name).name
            return ImagePayload(
                filename=filename or "source.bin",
                content_type=source.mime_type,
                data=payload,
            )

        path = Path(source.absolute_path)
        if not path.exists():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Stored source image is unavailable for regeneration.",
            )
        return ImagePayload(
            filename=path.name,
            content_type=source.mime_type,
            data=path.read_bytes(),
        )

    @staticmethod
    def _safe_remote_url(url: str) -> str:
        parts = urlsplit(url)
        safe_path = quote(parts.path, safe="/%._-~")
        safe_query = quote(parts.query, safe="=&%._-~")
        return urlunsplit((parts.scheme, parts.netloc, safe_path, safe_query, parts.fragment))
