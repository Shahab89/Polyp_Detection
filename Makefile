install_requirements:
	@pip install -r requirements.txt


streamlit:
	streamlit run polyp_detection/api/app.py

api_run:
	uvicorn polyp_detection.api.fast:app --reload

make_run:
	python polyp_detection/interface/main.py
