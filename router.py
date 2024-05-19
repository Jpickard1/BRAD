from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import HuggingFaceEncoder

routeGget = Route(
    name = 'GGET',
    utterances = read_prompts('routers/enrichr.txt')
)
routeScrape = Route(
    name = 'SCRAPE',
    utterances = read_prompts('routers/scrape.txt')
)
routeRAG = Route(
    name = 'RAG',
    utterances = read_prompts('routers/rag.txt')
)
routeLoad = Route(
    name = 'LOAD',
    utterances = read_prompts('routers/load.txt')
)

def getRouter():
    encoder = HuggingFaceEncoder()
    routes = [routeGget, routeScrape, routeLoad, routeRAG]
    router = RouteLayer(encoder=encoder, routes=routes)    
    return router

routeData = Route(
    name = 'data',
    utterances = read_prompts('routers/data.txt')
)

def getTableRouter():
    encoder = HuggingFaceEncoder()
    routes = [routeData]
    router = RouteLayer(encoder=encoder, routes=routes)    
    return router