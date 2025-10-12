#!/usr/bin/env python3
"""
Real-time event subscriber for your object detection events
"""
import requests
from requests.auth import HTTPDigestAuth
import sys
import json

def subscribe_to_events(camera_ip, username="root", password="pass"):
    url = f"http://{camera_ip}/axis-cgi/event.cgi"
    
    # Subscribe to your specific VideoAnalytics events
    params = {
        'action': 'subscribe',
        'events': 'tns1:VideoAnalytics/*'  # All VideoAnalytics events
    }
    
    print(f"🔗 Subscribing to VideoAnalytics events on {camera_ip}")
    print("🎯 Listening for events... (Press Ctrl+C to stop)")
    print("=" * 60)
    
    try:
        response = requests.get(url, params=params, 
                               auth=HTTPDigestAuth(username, password), 
                               stream=True, timeout=60)
        response.raise_for_status()
        
        for line in response.iter_lines(decode_unicode=True):
            if line:
                print(f"📨 {line}")
                
                # Parse JSON events
                if line.startswith('data:'):
                    try:
                        json_str = line[5:].strip()
                        if json_str:
                            event_data = json.loads(json_str)
                            
                            # Extract your event data
                            topic = event_data.get('topic', 'Unknown')
                            data = event_data.get('data', {})
                            timestamp = event_data.get('@timestamp', 'Unknown')
                            
                            if 'VideoAnalytics' in topic and 'ObjectDetected' in topic:
                                print(f"🎯 OBJECT DETECTION EVENT!")
                                print(f"   ⏰ Time: {timestamp}")
                                print(f"   📂 Topic: {topic}")
                                print(f"   🏷️  Class: {data.get('ObjectClass', 'N/A')}")
                                print(f"   📊 Confidence: {data.get('Confidence', 'N/A')}")
                                print(f"   📐 BBox: {data.get('BoundingBox', 'N/A')}")
                                print("-" * 60)
                                
                    except json.JSONDecodeError:
                        pass
                        
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python3 realtime_events.py <CAMERA_IP>")
    #     sys.exit(1)

    camera_ip = "169.254.118.229"
    subscribe_to_events(camera_ip)