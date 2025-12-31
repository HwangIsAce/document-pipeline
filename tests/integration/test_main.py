import os
import pytest
from unittest.mock import patch, MagicMock
import cocoindex

def test_main_initialization_logic():
    """main() 함수의 초기화 로직만 테스트"""
    from src.infrastructure.config import Config
    from src.application.container import ApplicationContainer
    
    config = Config()
    
    if not os.getenv("COCOINDEX_DATABASE_URL"):
        os.environ["COCOINDEX_DATABASE_URL"] = config.COCOINDEX_DATABASE_URL
        
    cocoindex.init()
    
    container = ApplicationContainer(config)
    
    # 초기화가 성공했는지 확인
    assert container is not None
    assert container.basic_pipeline is not None
    assert container.search_service is not None