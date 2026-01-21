import boto3

def verify_bedrock_access():
    """List available Bedrock models to verify access."""
    client = boto3.client('bedrock', region_name='us-east-1')
    
    response = client.list_foundation_models()
    
    print("✅ Bedrock Access Verified!")
    print(f"   Available models: {len(response['modelSummaries'])}")
    print("\nRecommended models for demos:")
    
    demo_models = [
        'amazon.nova-2-lite-v1:0',
        'openai.gpt-oss-20b-1:0'
        'anthropic.claude-3-haiku'
    ]
    
    for model in response['modelSummaries']:
        for demo in demo_models:
            if demo in model['modelId']:
                print(f"   ✓ {model['modelId']}")

if __name__ == "__main__":
    verify_bedrock_access()