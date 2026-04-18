import boto3
import json
from botocore.exceptions import ClientError

def generate_explanation(user_inputs: dict, prediction_result: str) -> str:
    if not user_inputs:
        return "No specific risk factors were identified for this analysis."

    prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful medical assistant explaining the clinical predictive results for a patient's cervical cancer risk.
<|eot_id|><|start_header_id|>user<|end_header_id|>
The patient's clinical data inputs are:
{json.dumps(user_inputs, indent=2)}

The model has predicted the patient's clinical risk level as: **{prediction_result} risk**.

Please generate a detailed, easy-to-understand explanation for the patient. 
1. Start by summarizing the prediction result ({prediction_result} risk) and briefly explaining what that means.
2. Underneath, explain EACH of the clinical data inputs provided by the patient one by one.
   - You MUST ensure there is an empty line (a double line break) between each individual factor's explanation.
   - Start each factor with a relevant emoji.
   - Put the input factor name in bold.
   - Provide a clear 1-2 sentence explanation of how this specific factor contributes to cervical cancer risk.

Do NOT mention any risk factors that were not explicitly provided in the JSON data above. 
End with a brief disclaimer about consulting a healthcare professional.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    body = {
        "prompt": prompt,
        "max_gen_len": 800,  # Increased for all factors
        "temperature": 0.5,
        "top_p": 0.9
    }

    try:
        bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1"
        )

        response = bedrock.invoke_model(
            modelId="meta.llama3-8b-instruct-v1:0",
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )

        response_body = json.loads(response["body"].read())
        return response_body.get("generation", "").strip()

    except ClientError as e:
        print(f"AWS Bedrock ClientError: {e}")
        return "AI explanation is currently unavailable."
    except Exception as e:
        print(f"General Error: {e}")
        return "An unexpected error occurred."
