import cv2
import numpy as np
import time
import csv

cap = cv2.VideoCapture(0)
# Initialize "Previous" values outside the loop
prev_x, prev_y, prev_theta, prev_time = 0, 0, 0, time.time()

measurements = []

while True:
    ret, frame = cap.read()
    if not ret: break
    
    curr_time = time.time()
    dt = max(curr_time - prev_time, 0.001) # Avoid division by zero
    
    # --- 1. INITIALIZE CURRENT COORDINATES ---
    # We set these to the previous values as a fallback
    cx, cy = prev_x, prev_y 
    px, py = 0, 0 
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- 2. TRACK THE CART (Green) ---
    lower_green = np.array([40, 70, 50])
    upper_green = np.array([80, 255, 255])
    mask_cart = cv2.inRange(hsv, lower_green, upper_green)
    conts_cart, _ = cv2.findContours(mask_cart, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if conts_cart:
        c = max(conts_cart, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.rectangle(frame, (cx-20, cy-20), (cx+20, cy+20), (0, 255, 0), 2)

    # --- 3. TRACK THE PENDULUM (Red) ---
    lower_red = np.array([0, 150, 50]) 
    upper_red = np.array([10, 255, 255])
    mask_pend = cv2.inRange(hsv, lower_red, upper_red)
    conts_pend, _ = cv2.findContours(mask_pend, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if conts_pend:
        p = max(conts_pend, key=cv2.contourArea)
        M_p = cv2.moments(p)
        if M_p["m00"] > 0:
            px = int(M_p["m10"] / M_p["m00"])
            py = int(M_p["m01"] / M_p["m00"])
            cv2.circle(frame, (px, py), 10, (0, 0, 255), -1)

    # --- 4. CALCULATE STATE (Now 'cy' is guaranteed to exist) ---
    # We only draw the 'rod' and calculate theta if both points were found
    if px != 0 and py != 0:
        cv2.line(frame, (cx, cy), (px, py), (0, 255, 0), 2)
        # Geometry: theta = atan2(opposite, adjacent)
        theta = np.arctan2(px - cx, cy - py) 
        
        # Calculate velocity
        v = (cx - prev_x) / dt
        omega = (theta - prev_theta) / dt
        
        # Store measurement row
        measurements.append({
            "time": curr_time,
            "cx": cx,
            "cy": cy,
            "px": px,
            "py": py,
            "theta": float(theta),
            "v": float(v),
            "omega": float(omega)
        })

        # Display Data
        cv2.putText(frame, f"Angle: {round(np.degrees(theta), 1)} deg", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Update previous values for next frame
        prev_x, prev_y, prev_theta, prev_time = cx, cy, theta, curr_time

    cv2.imshow("System Mapping", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if measurements:
    output_file = "measurements.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "cx", "cy", "px", "py", "theta", "v", "omega"])
        writer.writeheader()
        writer.writerows(measurements)
    print(f"Saved {len(measurements)} rows to {output_file}")
else:
    print("No measurements collected; nothing to save.")