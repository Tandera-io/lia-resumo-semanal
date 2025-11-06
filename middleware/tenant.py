"""
Middleware para detectar e validar tenant em cada requisição.
"""
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional
import logging
import httpx
import os

logger = logging.getLogger(__name__)

# Cache em memória (em produção, usar Redis)
_tenant_cache = {}


class TenantContext:
    """Contexto do tenant para a requisição atual."""
    
    def __init__(self):
        self.tenant_slug: Optional[str] = None
        self.tenant_data: Optional[dict] = None
    
    def set_tenant(self, slug: str, data: dict):
        self.tenant_slug = slug
        self.tenant_data = data
    
    def get_supabase_url(self) -> str:
        if not self.tenant_data:
            raise ValueError("Tenant não configurado")
        return self.tenant_data.get("supabaseUrl", "")
    
    def get_anon_key(self) -> str:
        if not self.tenant_data:
            raise ValueError("Tenant não configurado")
        return self.tenant_data.get("anonKey", "")
    
    def get_service_key(self) -> Optional[str]:
        """Service key não vem do endpoint público, apenas para referência."""
        if not self.tenant_data:
            raise ValueError("Tenant não configurado")
        return self.tenant_data.get("serviceRole")


# Contexto global por requisição (usando contextvars seria ideal em produção)
_tenant_context = TenantContext()


def get_tenant_context() -> TenantContext:
    """Retorna o contexto do tenant atual."""
    return _tenant_context


async def get_tenant_from_registry(slug: str) -> Optional[dict]:
    """
    Busca dados do tenant no Registry API.
    
    Args:
        slug: Slug do tenant (ex: 'acme-corp')
    
    Returns:
        Dados do tenant incluindo credenciais Supabase
    """
    # Verificar cache
    if slug in _tenant_cache:
        logger.debug(f"Tenant '{slug}' encontrado no cache")
        return _tenant_cache[slug]
    
    # Configurações do Registry
    registry_url = os.getenv("REGISTRY_API_URL", "http://localhost:3000")
    
    # Buscar no Registry
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Endpoint público para buscar tenant por slug
            url = f"{registry_url}/api/tenants/by-slug/{slug}"
            response = await client.get(url)
            
            if response.status_code == 404:
                logger.error(f"Tenant '{slug}' não encontrado no Registry")
                return None
            
            if response.status_code >= 400:
                logger.error(f"Erro ao buscar tenant: {response.status_code} - {response.text}")
                return None
            
            data = response.json()
            tenant_data = data.get("tenant", {})
            
            # Armazenar no cache
            _tenant_cache[slug] = tenant_data
            logger.info(f"Tenant '{slug}' carregado do Registry")
            
            return tenant_data
            
    except Exception as e:
        logger.error(f"Erro ao comunicar com Registry: {e}")
        return None


def clear_tenant_cache(slug: Optional[str] = None):
    """Limpa o cache de tenants."""
    global _tenant_cache
    if slug:
        _tenant_cache.pop(slug, None)
        logger.info(f"Cache do tenant '{slug}' limpo")
    else:
        _tenant_cache.clear()
        logger.info("Cache de todos os tenants limpo")


class TenantMiddleware(BaseHTTPMiddleware):
    """
    Middleware que detecta o tenant via:
    1. Header X-Tenant-Slug
    2. Subdomain (ex: acme.localhost)
    3. Query param ?tenant=slug
    """
    
    async def dispatch(self, request: Request, call_next):
        # Ignorar requisições OPTIONS (CORS preflight) - elas não têm headers customizados
        if request.method == "OPTIONS":
            logger.debug(f"[TenantMiddleware] Ignorando OPTIONS para {request.url.path}")
            response = await call_next(request)
            return response
        
        tenant_slug = None
        
        # 1. Tentar via header (prioritário)
        tenant_slug = request.headers.get("X-Tenant-Slug")
        
        # 2. Tentar via subdomain
        if not tenant_slug:
            host = request.headers.get("host", "")
            if "." in host and not host.startswith("localhost"):
                tenant_slug = host.split(".")[0]
        
        # 3. Tentar via query param
        if not tenant_slug:
            tenant_slug = request.query_params.get("tenant")
        
        # Se não encontrou tenant, usar tenant padrão (para compatibilidade)
        if not tenant_slug:
            tenant_slug = os.getenv("DEFAULT_TENANT_SLUG", "advice")
            logger.debug(f"Nenhum tenant especificado, usando '{tenant_slug}'")
        
        # Buscar dados do tenant no Registry
        try:
            tenant_data = await get_tenant_from_registry(tenant_slug)
            if not tenant_data:
                # Se não encontrou no Registry, deixar passar para usar .env (fallback)
                logger.warning(f"Tenant '{tenant_slug}' não encontrado, usando credenciais padrão")
                _tenant_context.set_tenant(tenant_slug, {})
            else:
                # Configurar contexto do tenant
                _tenant_context.set_tenant(tenant_slug, tenant_data)
                logger.info(f"Tenant configurado: {tenant_slug}")
            
        except Exception as e:
            logger.error(f"Erro ao buscar tenant '{tenant_slug}': {e}")
            # Continuar com credenciais padrão
            _tenant_context.set_tenant(tenant_slug, {})
        
        # Processar requisição
        response = await call_next(request)
        return response


# Manter função legada para compatibilidade
async def tenant_middleware(request: Request, call_next):
    """Função wrapper para compatibilidade com código legado."""
    middleware = TenantMiddleware(app=None)
    return await middleware.dispatch(request, call_next)
