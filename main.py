from fastapi import FastAPI, UploadFile, File, HTTPException
from prediction import read_image, preprocess, predict, load_model
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)


@app.on_event("startup")
async def startup_event():
    print("FastAPI startup event triggered. Loading model...")
    try:
       
        loaded_model_instance = load_model()
       
        app.state.model = loaded_model_instance
        print("Model stored in app.state successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Model failed to load during startup: {e}")
       
        raise 


@app.get("/")
def hello_world(name: str):
    return f"Hello {name}!"

@app.post("/api/predict")
async def model_predict(file: UploadFile = File(...)):
    model = app.state.model 

    if model is None:

         raise HTTPException(status_code=500, detail="TensorFlow model is not available (startup failed).")

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading uploaded file: {e}")

    try:
        pil_image = read_image(image_bytes)
    except Exception as e:
         raise HTTPException(status_code=400, detail=f"Could not open image file: {e}")

    try:
        processed_image_array = preprocess(pil_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during image preprocessing: {e}")

    try:
        prediction_data = predict(model, processed_image_array) 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {e}")


    class_label_mapping = {
        0: 'no',
        1: 'yes',
    }

    try:
        predicted_label = class_label_mapping[prediction_data]
        print(f"Predicted label (on server): {predicted_label}")
    except KeyError:
         raise HTTPException(status_code=500, detail=f"Model returned unexpected class index: {prediction_data}")
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error mapping prediction to label: {e}")


    return {"prediction": predicted_label}
