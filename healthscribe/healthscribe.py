import boto3
from datetime import datetime
import time
import requests
import json
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()  # Load environment variables from .env file

# AWS clients
transcribe_medical = boto3.client('transcribe', region_name=os.getenv('AWS_DEFAULT_REGION'))
s3_client = boto3.client('s3', region_name=os.getenv('AWS_DEFAULT_REGION'))
brt = boto3.client("bedrock-runtime", region_name=os.getenv('AWS_DEFAULT_REGION'))

# Default settings from environment variables
BUCKET_NAME = os.getenv('BUCKET_NAME')
DATA_ACCESS_ROLE_ARN = os.getenv('DATA_ACCESS_ROLE_ARN')
# Allowed extensions for audio files
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a'}

# Global variable to store transcription summary
transcription_summary = None

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def upload_to_s3(file_path, filename):
    """Uploads a local audio file to S3 and returns its S3 URL"""
    try:
        s3_key = f"health_scribe/uploads/{filename}"
        s3_client.upload_file(file_path, BUCKET_NAME, s3_key)
        file_url = f"s3://{BUCKET_NAME}/{s3_key}"
        print(f"DEBUG: File uploaded to S3: {file_url}")
        return file_url
    except Exception as e:
        raise Exception(f"Error uploading file to S3: {str(e)}")

def generate_presigned_url(bucket_name, object_key, expiration=3600):
    """
    Generate a pre-signed URL for summary.json to allow temporary public access.
    The URL expires in 'expiration' seconds (default: 1 hour).
    """
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_key},
            ExpiresIn=expiration  # URL expires in 1 hour
        )
        print(f"Generated pre-signed URL: {url}")
        return url
    except Exception as e:
        print(f"Error generating pre-signed URL: {str(e)}")
        return None


def fetch_summary(summary_uri):
    """
    Fetches the summary.json file using a pre-signed URL and formats it into plain text.
    """
    try:
        # Extract the S3 object key from the URI
        object_key = summary_uri.split(f"{BUCKET_NAME}/")[-1]

        # Generate a pre-signed URL for temporary access
        pre_signed_url = generate_presigned_url(BUCKET_NAME, object_key)

        # Fetch the summary.json file from the pre-signed URL
        response = requests.get(pre_signed_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch summary.json: {response.status_code}, {response.text}")

        summary_json = response.json()

        # Parse the JSON to extract summarized text
        summary_text = ""
        sections = summary_json.get("ClinicalDocumentation", {}).get("Sections", [])
        for section in sections:
            section_name = section.get("SectionName", "Unknown Section")
            summary_text += f"\n{section_name}:\n"
            for summary in section.get("Summary", []):
                summarized_segment = summary.get("SummarizedSegment", "")
                summary_text += f"- {summarized_segment}\n"

        return summary_text.strip()

    except Exception as e:
        raise Exception(f"Error fetching summary: {str(e)}")
    

def start_transcription(job_name, audio_file_uri):
    """Starts a transcription job for the provided audio file URL"""
    print("----",job_name)
    print("----", audio_file_uri)
    try:
        existing_jobs = transcribe_medical.list_medical_scribe_jobs(Status='IN_PROGRESS', MaxResults=5)
        print(existing_jobs)
        active_jobs = existing_jobs.get('MedicalScribeJobSummaries', [])
        if active_jobs:
            active_job = active_jobs[0]
            return poll_transcription_job(active_job['MedicalScribeJobName'])
    except Exception as e:
        raise Exception(f"Error checking active transcription jobs: {e}")

    try:
        transcribe_medical.start_medical_scribe_job(
            MedicalScribeJobName=job_name,
            Media={'MediaFileUri': audio_file_uri},
            OutputBucketName=BUCKET_NAME,
            DataAccessRoleArn=DATA_ACCESS_ROLE_ARN,
            Settings={'ShowSpeakerLabels': True, 'MaxSpeakerLabels': 2}
        )
    except Exception as e:
        raise Exception(f"Error starting transcription job: {e}")

    return poll_transcription_job(job_name)


def poll_transcription_job(job_name):
    """Polls the transcription job status until it is completed or failed"""
    while True:
        try:
            response = transcribe_medical.get_medical_scribe_job(MedicalScribeJobName=job_name)
            status = response['MedicalScribeJob']['MedicalScribeJobStatus']
            if status == 'COMPLETED':
                return response['MedicalScribeJob']['MedicalScribeOutput']
            elif status == 'FAILED':
                raise Exception(f"Job '{job_name}' failed.")
            time.sleep(15)
        except Exception as e:
            raise Exception(f"Error checking job status: {e}")

def ask_claude(question, summary):
   """
   Queries Claude using the Conversation API.
   """
   conversation = [
       {
           "role": "user",
           "content": [{"text": f"Here is the summary of the medical transcription:\n{json.dumps(summary, indent=2)}\n\nNow, based on this summary, please answer the following question:\n{question}"}]
       }
   ]

   try:
       response = brt.converse(
           modelId="arn:aws:bedrock:us-east-1:992382417943:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # ✅ Use the model you have access to
           messages=conversation,
           inferenceConfig={
               "maxTokens": 512,
               "temperature": 0.5,
               "topP": 0.9
           }
       )
       response_text = response["output"]["message"]["content"][0]["text"]
       return response_text
   except Exception as e:
       raise Exception(f"Error querying Claude: {e}")
