import time
import requests
def get_light_value():
    try:
        response = requests.get("http://localhost:3000/light")  
        response.raise_for_status()
        data = response.json()
        return data.get("light", 0) 
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def main():
    while True:
        light_value = get_light_value()
        if light_value is not None:
            print(f"Current light value: {light_value}")
            if light_value < 50 :  
                print("intensity")
        
        time.sleep(1)  

if __name__ == "__main__":
    main()
