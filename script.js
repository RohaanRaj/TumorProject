document.getElementById('predictButton').addEventListener('click', async () => {
const imageInput =document.getElementById('imageInput');
const resultArea= document.getElementById('resultArea');
const file = imageInput.files[0];
if(!file){
    predictionText.textContent='Please select image.';
    return;
}
const formData= new FormData();
formData.append('file',file);
predictionText.textContent = 'Predicting.....';

try{
    const response = await fetch('http://54.242.224.249:8000/api/predict',{
        method: 'POST',
        body: formData
    });

    if (response.ok){
        const result = await response.json();
        //predictionText.textContent=`Prediction: ${result.prediction}`;

        if (result.prediction === 'yes') {
            predictionText.textContent=`Tumor found! You're so Dead bro`;
            resultImage.src = 'images/depressed.png';
        } else if (result.prediction === 'no') {
            predictionText.textContent=`No tumor you're fine cutie`;
            resultImage.src = 'images/kiss2.jpeg';
        }
        resultImage.style.display = 'block';

    } else{
        const error = await response.json();
        predictionText.textContent= `Error :${error.detail || response.statusText}`;
    }
    }
    catch(error){
        predictionText.textContent=`An error occured :${error.message}`;
    }


});
