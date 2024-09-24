// Adding event listener to the 'Generate' button
document.getElementById('generate').addEventListener('click', async function () {
    // Get user input values
    const token = document.getElementById('token').value;
    const prompt = document.getElementById('prompt').value;
    const steps = document.getElementById('steps').value;
    const guidance = document.getElementById('guidance').value;
    const gptModel = document.getElementById('gpt').value;

    // Validate token input
    if (!token) {
        alert("Please enter your Hugging Face Token.");
        return;
    }

    // Show spinner and hide previous results
    document.getElementById('spinner').style.display = 'block';
    document.getElementById('result').style.display = 'none';

    // Record start time for processing
    const startTime = performance.now();

    try {
        // Fetch image from Hugging Face API
        const response = await fetch('https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                inputs: prompt,
                parameters: {
                    num_inference_steps: parseInt(steps),
                    guidance_scale: parseFloat(guidance),
                }
            })
        });

        // Check if the response is not okay
        if (!response.ok) {
            throw new Error('Failed to generate image.');
        }

        // Convert response into an image blob
        const imageBlob = await response.blob();
        const imageUrl = URL.createObjectURL(imageBlob);

        // Display the generated image
        document.getElementById('image').src = imageUrl;
        document.getElementById('result').style.display = 'block';

        // Calculate and display processing time
        const endTime = performance.now();
        const processingTime = ((endTime - startTime) / 1000).toFixed(2);
        document.getElementById('time').textContent = `Processing time: ${processingTime} seconds`;

    } catch (error) {
        // Show error message in case of failure
        alert("Error generating image. Please check your token or internet connection.");
        console.error(error);
    }

    // Hide spinner once the process is done
    document.getElementById('spinner').style.display = 'none';
});
