@echo off
setlocal enabledelayedexpansion
title Large-Scale Social Media Intelligence Platform - Launcher

cd /d "%~dp0"

echo ====================================================================
echo   Large-Scale Social Media Intelligence Platform
echo   Track: Data Science
echo   Tech Stack: Reddit/YouTube APIs, Spark NLP, BERTopic,
echo               NetworkX, Neo4j, Elasticsearch, Kibana
echo ====================================================================
echo.

:: 1. Locate Python environment
set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    echo [!] Virtual environment not detected at .venv\Scripts\python.exe
    echo Attempting to initialize environment using uv...
    where uv >nul 2>nul
    if %errorlevel% equ 0 (
        echo Creating Python 3.11 environment with uv...
        uv venv --python 3.11
        uv pip install -r "%~dp0requirements.txt"
    ) else (
        echo Error: Neither .venv nor uv was found. Please ensure Python 3.11 is installed.
        pause
        exit /b 1
    )
)

:: If a command line parameter was passed (e.g., run.bat --run-all or run.bat --test)
if not "%~1"=="" (
    if "%~1"=="--test" (
        goto RUN_TESTS
    ) else (
        goto RUN_CUSTOM
    )
)

:MENU
echo.
echo Please select an operation:
echo   [1] Run Full End-to-End Pipeline (Steps 1 to 7) [RECOMMENDED]
echo   [2] Run Complete Automated Test Suite (21 Tests)
echo   [3] Step 1: Collect Data (Reddit ^& YouTube APIs)
echo   [4] Step 2: Clean Text, Handle Emojis, Detect Language
echo   [5] Step 3: Spark NLP Multilingual Sentiment Tracking
echo   [6] Step 4: BERTopic Topic Modeling (Rise ^& Fall)
echo   [7] Step 5: Neo4j Graph ^& NetworkX PageRank Ranking
echo   [8] Step 6: Train LIAR Misinformation Classifier
echo   [9] Step 7: Elasticsearch Indexing ^& Kibana Dashboard
echo   [0] Exit
echo.
set /p CHOICE="Enter your choice [0-9] (Default is 1): "

if "%CHOICE%"=="" set CHOICE=1
if /i "%CHOICE%"=="1" goto RUN_ALL
if /i "%CHOICE%"=="2" goto RUN_TESTS
if /i "%CHOICE%"=="3" goto RUN_STEP1
if /i "%CHOICE%"=="4" goto RUN_STEP2
if /i "%CHOICE%"=="5" goto RUN_STEP3
if /i "%CHOICE%"=="6" goto RUN_STEP4
if /i "%CHOICE%"=="7" goto RUN_STEP5
if /i "%CHOICE%"=="8" goto RUN_STEP6
if /i "%CHOICE%"=="9" goto RUN_STEP7
if /i "%CHOICE%"=="0" goto EXIT_PROMPT

echo Invalid choice. Please try again.
goto MENU

:RUN_ALL
echo.
echo ====================================================================
echo   Executing Full 7-Step Intelligence Pipeline...
echo ====================================================================
"%VENV_PYTHON%" main.py --run-all
goto FINISH

:RUN_TESTS
echo.
echo ====================================================================
echo   Running All 21 Automated Unit ^& Integration Tests...
echo ====================================================================
"%VENV_PYTHON%" -m unittest discover -s tests -p "test_*.py" -v
goto FINISH

:RUN_STEP1
echo.
echo ====================================================================
echo   Step 1: Collecting Data from Reddit and YouTube APIs...
echo ====================================================================
"%VENV_PYTHON%" data_collection\collector_runner.py --sources reddit youtube --limit 15
goto FINISH

:RUN_STEP2
echo.
echo ====================================================================
echo   Step 2: Cleaning Text, Handling Emojis ^& Detecting Language...
echo ====================================================================
"%VENV_PYTHON%" preprocessing\text_cleaner.py
goto FINISH

:RUN_STEP3
echo.
echo ====================================================================
echo   Step 3: Spark NLP Multilingual Sentiment Analysis ^& Tracking...
echo ====================================================================
"%VENV_PYTHON%" sentiment_analysis\sentiment_analyzer.py
goto FINISH

:RUN_STEP4
echo.
echo ====================================================================
echo   Step 4: BERTopic Topic Modeling ^& Dynamic Trajectories...
echo ====================================================================
"%VENV_PYTHON%" topic_modeling\topic_modeler.py
goto FINISH

:RUN_STEP5
echo.
echo ====================================================================
echo   Step 5: Neo4j Graph ^& NetworkX PageRank Influencer Ranking...
echo ====================================================================
"%VENV_PYTHON%" graph_analysis\networkx_analyzer.py
goto FINISH

:RUN_STEP6
echo.
echo ====================================================================
echo   Step 6: Training Misinformation Classifier on LIAR Dataset...
echo ====================================================================
"%VENV_PYTHON%" misinformation_detection\misinformation_classifier.py
goto FINISH

:RUN_STEP7
echo.
echo ====================================================================
echo   Step 7: Elasticsearch Bulk Indexing ^& Kibana Dashboard Objects...
echo ====================================================================
"%VENV_PYTHON%" indexing_and_dashboard\elasticsearch_indexer.py
goto FINISH

:RUN_CUSTOM
"%VENV_PYTHON%" main.py %*
goto FINISH

:FINISH
set "EXIT_CODE=%errorlevel%"
echo.
echo ====================================================================
echo   Execution Finished with exit code: %EXIT_CODE%
echo   Processed Data Artifacts (data\processed\):
echo     - clean_posts.json
echo     - posts_with_sentiment.json ^& sentiment_trends.json
echo     - posts_with_topics.json ^& topics_over_time.json
echo     - posts_with_graph_metrics.json ^& influencer_rankings.json
echo     - posts_with_misinformation.json
echo     - elasticsearch_bulk.ndjson
echo ====================================================================
echo.
if not "%~1"=="" exit /b %EXIT_CODE%
pause
goto MENU

:EXIT_PROMPT
echo Exiting...
exit /b 0
