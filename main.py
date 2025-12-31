# main.py
import os
import uvicorn
import cocoindex

from src.infrastructure.config import Config
from src.application.container import ApplicationContainer
from src.infrastructure.api.app import create_app

def main():
    config = Config()
    
    if not os.getenv("COCOINDEX_DATABASE_URL"):
        os.environ["COCOINDEX_DATABASE_URL"] = config.COCOINDEX_DATABASE_URL
        
    cocoindex.init()

    container = ApplicationContainer(config)

    container.qdrant_provider.get_connection()
    
    cocoindex.setup_all_flows(report_to_stdout=False)

    app = create_app(container)
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()