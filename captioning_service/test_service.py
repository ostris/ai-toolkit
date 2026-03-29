#!/usr/bin/env python3
"""
Test script for the JoyCaption service
This script tests the captioning service without requiring GPU/model loading
"""

import json
import requests
import time
import sys
from pathlib import Path

def test_health_endpoint():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get("http://127.0.0.1:5000/health", timeout=5)
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Service status: {data.get('status')}")
            print(f"Model loaded: {data.get('model_loaded')}")
            print(f"GPU available: {data.get('gpu_available')}")
            print(f"GPU count: {data.get('gpu_count')}")
            return True
        else:
            print(f"Health check failed: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to service: {e}")
        return False

def test_prompts_endpoint():
    """Test the prompts endpoint"""
    print("\nTesting prompts endpoint...")
    try:
        response = requests.get("http://127.0.0.1:5000/prompts", timeout=5)
        print(f"Prompts status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Available styles: {data.get('styles')}")
            return True
        else:
            print(f"Prompts check failed: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Failed to get prompts: {e}")
        return False

def test_caption_endpoint():
    """Test the caption endpoint with a sample image"""
    print("\nTesting caption endpoint...")
    
    # Check if test image exists
    test_image = Path("../datasets/test_dataset/test_image.jpg")
    if not test_image.exists():
        print(f"Test image not found: {test_image}")
        return False
    
    try:
        payload = {
            "image_path": str(test_image.absolute()),
            "style": "descriptive",
            "max_new_tokens": 100,
            "temperature": 0.6
        }
        
        response = requests.post(
            "http://127.0.0.1:5000/caption", 
            json=payload, 
            timeout=30
        )
        
        print(f"Caption status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Caption success: {data.get('success')}")
            if data.get('success'):
                print(f"Generated caption: {data.get('caption')}")
                print(f"Generation time: {data.get('generation_time'):.2f}s")
            return True
        else:
            print(f"Caption failed: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Failed to caption image: {e}")
        return False

def main():
    """Run all tests"""
    print("JoyCaption Service Test Suite")
    print("=" * 40)
    
    # Test if service is running
    if not test_health_endpoint():
        print("\n❌ Service is not running or not responding")
        print("To start the service, run: ./start_service.sh")
        return False
    
    # Test prompts endpoint
    if not test_prompts_endpoint():
        print("\n❌ Prompts endpoint failed")
        return False
    
    # Test caption endpoint (only if model is loaded)
    print("\nChecking if model is loaded...")
    try:
        health_response = requests.get("http://127.0.0.1:5000/health")
        health_data = health_response.json()
        
        if health_data.get('model_loaded'):
            print("✅ Model is loaded, testing caption generation...")
            if not test_caption_endpoint():
                print("\n❌ Caption endpoint failed")
                return False
        else:
            print("⚠️  Model not loaded, skipping caption test")
            print("Note: The service may still be loading the model")
    except:
        print("⚠️  Could not check model status")
    
    print("\n✅ All available tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
