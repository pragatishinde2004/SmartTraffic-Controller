import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load YOLO model files
weights_path = "yolo_vehicle_detection/yolov3.weights"
config_path = "yolo_vehicle_detection/yolov3.cfg.txt"
labels_path = "yolo_vehicle_detection/coco.names.txt"

# Load class labels
with open(labels_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
    
# Add "police car" if missing
if "police car" not in classes:
    classes.append("police car")

# Initialize YOLO network
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Constants
VEHICLE_THRESHOLD = 10  # Vehicles needed to trigger signal change
EMERGENCY_CLASSES = ["ambulance", "firetruck", "police car"]

def detect_vehicles(image_path, visualize=False):
    """Process an image and return vehicle count + emergency status"""
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Could not load image: {image_path}")
        return 0, False

    height, width, _ = img.shape
    
    # YOLO detection
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Detection processing
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                boxes.append([int(center_x - w/2), int(center_y - h/2), w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Count vehicles
    vehicle_count = 0
    emergency_detected = False
    
    for i in indices.flatten():
        label = classes[class_ids[i]]
        
        if label in ["car", "bus", "truck"]:
            vehicle_count += 1
            color = (0, 255, 0)
        elif label in EMERGENCY_CLASSES:
            emergency_detected = True
            color = (0, 0, 255)
            logging.warning(f"EMERGENCY VEHICLE DETECTED: {label}")
        else:
            color = (255, 255, 0)
            
        if visualize:
            x, y, w, h = boxes[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if visualize:
        cv2.imshow('Detections', img)
        cv2.waitKey(500)  # Display for 500ms
        cv2.destroyAllWindows()

    return vehicle_count, emergency_detected

def control_signals(lane1, lane2):
    """Determine traffic signals based on both lanes' status"""
    # Emergency priority
    if lane1['emergency']:
        return "GREEN", "RED"
    if lane2['emergency']:
        return "RED", "GREEN"
    
    # Vehicle count logic
    if lane1['count'] > VEHICLE_THRESHOLD and lane2['count'] <= VEHICLE_THRESHOLD:
        return "GREEN", "RED"
    elif lane2['count'] > VEHICLE_THRESHOLD and lane1['count'] <= VEHICLE_THRESHOLD:
        return "RED", "GREEN"
    
    # Default case (both below threshold or both above)
    return "YELLOW", "YELLOW"

def main():
    # Input images
    lane1_img = "images/sample_image1.png"
    lane2_img = "images/sample_image2.png"

    # Process both lanes
    lane1_count, lane1_emergency = detect_vehicles(lane1_img, visualize=True)
    lane2_count, lane2_emergency = detect_vehicles(lane2_img, visualize=True)

    lane1_status = {'count': lane1_count, 'emergency': lane1_emergency}
    lane2_status = {'count': lane2_count, 'emergency': lane2_emergency}

    # Determine signals
    lane1_signal, lane2_signal = control_signals(lane1_status, lane2_status)
    
    # Output results
    print("\n=== TRAFFIC SIGNAL STATUS ===")
    print(f"Lane 1: {lane1_signal} | Vehicles: {lane1_count}")
    print(f"Lane 2: {lane2_signal} | Vehicles: {lane2_count}")
    print("============================\n")

    # Simulate signal change (replace with actual GPIO/API calls)
    if lane1_signal == "GREEN":
        print("Activating Lane 1 Green Light")
    if lane2_signal == "GREEN":
        print("Activating Lane 2 Green Light")

if __name__ == "__main__":
    main()
