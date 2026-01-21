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

    container.default_db_provider.get_connection()
    
    try:
        cocoindex.setup_all_flows(report_to_stdout=False)
    except RuntimeError as e:
        # 컬렉션이 이미 존재하는 경우 에러를 무시
        error_msg = str(e)
        if "already exists" in error_msg:
            print("Collections already exist, skipping setup...")
        else:
            raise

    app = create_app(container)
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()