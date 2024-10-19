from picrawler import PiCrawler
from robot_hat import TTS, Music, Ultrasonic, Pin, Camera
import time
import math
import csv
import matplotlib.pyplot as plt
import random
import logging
import cv2
import numpy as np

# Initialize modules
tts = TTS()
music = Music()
crawler = PiCrawler([10, 11, 12, 4, 5, 6, 1, 2, 3, 7, 8, 9])
sonar = Ultrasonic(Pin("D2"), Pin("D3"))
camera = Camera()

# Constants
ALERT_DISTANCE = 15  # Distance in cm to trigger obstacle avoidance
BASE_SPEED = 100  # Base speed for movement
TURN_ANGLE = 90  # Default turn angle when avoiding obstacles
PHOTO_INTERVAL = 10  # Take a photo every 10 seconds

# Data storage for mapping
data = []
photo_history = []  # Store previous photos for orientation

# Set up logging
logging.basicConfig(level=logging.INFO)

def measure_distance():
    """Measure the distance using the ultrasonic sensor."""
    distance = sonar.read()
    if distance is None or distance < 0:
        return float('inf')  # Return a very large value if no valid reading
    return distance

def save_data_to_csv(data):
    """Save recorded angle and distance data to a CSV file."""
    with open('room_map.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Angle", "Distance"])
        writer.writerows(data)

def save_plot(data):
    """Save a scatter plot of the room map."""
    x_coords = [d * math.cos(math.radians(a)) for a, d in data]
    y_coords = [d * math.sin(math.radians(a)) for a, d in data]
    plt.figure(figsize=(10, 10))
    plt.plot(x_coords, y_coords, 'bo-')  # Blue circles connected by lines
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.title('Room Map')
    plt.grid(True)
    plt.savefig('room_map.png')
    plt.close()

def capture_photo():
    """Capture a photo using the camera module."""
    photo = camera.read()
    return photo

def compare_photos(photo1, photo2):
    """Compare two photos to determine if they are similar."""
    # Convert images to grayscale
    gray1 = cv2.cvtColor(photo1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(photo2, cv2.COLOR_BGR2GRAY)
    
    # Use ORB (Oriented FAST and Rotated BRIEF) to detect and compute features
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    # Use BFMatcher to find matches between the descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Consider the photos similar if a significant number of matches are close
    num_good_matches = len([m for m in matches if m.distance < 50])
    return num_good_matches > 10  # Threshold for considering photos similar

def wander():
    """Make the robot wander randomly in the environment."""
    action = random.choice(['forward', 'turn left', 'turn right'])
    
    if action == 'forward':
        logging.info("Wandering forward.")
        crawler.do_action('forward', 1, BASE_SPEED)
    elif action == 'turn left':
        logging.info("Turning left randomly.")
        crawler.do_action('turn left angle', TURN_ANGLE, BASE_SPEED)
    elif action == 'turn right':
        logging.info("Turning right randomly.")
        crawler.do_action('turn right angle', TURN_ANGLE, BASE_SPEED)
    
    time.sleep(0.2)  # Adjust based on movement

def avoid_obstacle():
    """Make the robot avoid obstacles by turning away from them."""
    logging.info("Obstacle detected, turning left to avoid.")
    crawler.do_action('turn left angle', TURN_ANGLE, BASE_SPEED)
    time.sleep(0.2)  # Adjust based on movement

def check_orientation(current_photo):
    """Check if the robot is in a previously visited area using photo comparison."""
    for previous_photo in photo_history:
        if compare_photos(current_photo, previous_photo):
            logging.info("Recognized a previously visited area.")
            # Take some action, e.g., avoid revisiting this area or turn in a new direction
            avoid_obstacle()
            return True
    return False

def main():
    last_photo_time = time.time()
    
    while True:
        try:
            distance = measure_distance()
            logging.info(f"Distance measured: {distance} cm")
            
            # Adjust speed dynamically based on the distance from obstacles
            speed = max(50, int(distance / ALERT_DISTANCE * BASE_SPEED))
            
            if distance < ALERT_DISTANCE:
                # Obstacle detected within alert distance, take evasive action
                try:
                    music.sound_effect_threading('./sounds/sign.wav')
                except Exception as e:
                    logging.error(f"Error playing sound: {e}")
                avoid_obstacle()
            else:
                # No obstacle detected, continue wandering
                wander()
            
            # Take a photo every PHOTO_INTERVAL seconds and check for orientation
            if time.time() - last_photo_time > PHOTO_INTERVAL:
                current_photo = capture_photo()
                if current_photo is not None:
                    if not check_orientation(current_photo):
                        photo_history.append(current_photo)
                last_photo_time = time.time()
            
            # Record the distance and angle
            # Angle calculation would typically depend on turning actions, so we assume random angles here.
            angle = random.randint(0, 360)  # In reality, this would depend on movement
            data.append((angle, distance))
        
        except KeyboardInterrupt:
            logging.info("Program interrupted, saving data.")
            save_data_to_csv(data)
            save_plot(data)
            logging.info("Data saved to room_map.csv and room_map.png")
            break

if __name__ == "__main__":
    main()
