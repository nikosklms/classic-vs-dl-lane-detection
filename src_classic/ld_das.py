import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    """
    Υπολογίζει τις συντεταγμένες (x1, y1, x2, y2) της γραμμής
    βασισμένο στην κλίση (slope) και το σημείο τομής (intercept).
    """
    try:
        slope, intercept = line_parameters
    except TypeError:
        return np.array([0, 0, 0, 0])
        
    y1 = image.shape[0] # Ξεκινάει από το κάτω μέρος της εικόνας
    y2 = int(y1 * (3/5)) # Σταματάει στα 3/5 του ύψους (λίγο πάνω από τη μέση)
    
    # y = mx + b  -->  x = (y - b) / m
    # Προστασία από διαίρεση με το μηδέν
    if slope == 0:
        slope = 0.001
        
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    """
    Υπολογίζει τον μέσο όρο των αριστερών και δεξιών γραμμών
    για να δημιουργήσει δύο ενιαίες γραμμές λωρίδων.
    """
    left_fit = []
    right_fit = []
    
    if lines is None:
        return None
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        
        # Fit a polynomial (degree 1 -> ευθεία γραμμή y = mx + b)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        
        # Φιλτράρισμα θορύβου (οριζόντιες γραμμές)
        if abs(slope) < 0.3:
            continue
            
        # Αριστερή λωρίδα (αρνητική κλίση) vs Δεξιά λωρίδα (θετική κλίση)
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
            
    # Υπολογισμός μέσου όρου
    lines_to_show = []
    
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        lines_to_show.append(make_coordinates(image, left_fit_average))
        
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        lines_to_show.append(make_coordinates(image, right_fit_average))
        
    if not lines_to_show:
        return None
        
    return np.array(lines_to_show)

def canny(image):
    """
    Μετατροπή σε Grayscale, Blur και Canny Edge Detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_img = cv2.Canny(blur, 40, 120)
    return canny_img

def display_lines(image, lines):
    """
    Ζωγραφίζει τις γραμμές πάνω σε μια μαύρη εικόνα.
    """
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Μπλε γραμμές (255, 0, 0) με πάχος 10
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(image):
    """
    Δημιουργεί μάσκα για να εστιάσουμε μόνο στο δρόμο.
    Προσαρμοσμένο από το repo για να δουλεύει με ποσοστά (dynamic size)
    ώστε να μην κρασάρει αν αλλάξει η ανάλυση της εικόνας.
    """
    height = image.shape[0]
    width = image.shape[1]
    
    polygons = np.array([
        [
            (int(width * 0.0), height),          # Bottom Left
            (int(width * 1.0), height),          # Bottom Right
            (int(width * 0.67), int(height * 0.37)), # Top Right
            (int(width * 0.4), int(height * 0.37))  # Top Left
        ]
    ], dtype=np.int32)
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# --- Main Execution Loop ---
if __name__ == "__main__":
    img_dir = "images_hybrid/"
    out_dir = "out/"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created directory: {out_dir}")

    if os.path.exists(img_dir):
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()
        
        if image_files:
            print(f"Found {len(image_files)} images. Processing...")
            
            for img_file in image_files:
                img_path = os.path.join(img_dir, img_file)
                frame = cv2.imread(img_path)
                
                if frame is not None:
                    lane_image = np.copy(frame)
                    
                    # --- PATH A: Lane Detection (Your Original Logic) ---
                    canny_image = canny(lane_image)
                    cropped_canny = region_of_interest(canny_image)
                    lines = cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=15)
                    averaged_lines = average_slope_intercept(lane_image, lines)
                    line_image = display_lines(lane_image, averaged_lines)
                    
                    # --- PATH B: Drivable Area (New Divergent Path) ---
                    #Convert to HSV to isolate road color
                    hsv = cv2.cvtColor(lane_image, cv2.COLOR_BGR2HSV)
                    
                    #Define range for gray asphalt (Adjust these values if needed)
                    lower_gray = np.array([0, 0, 50])   
                    upper_gray = np.array([180, 50, 200])
                    
                    #Create mask and apply ROI
                    drivable_mask = cv2.inRange(hsv, lower_gray, upper_gray)
                    
                    # Morphological ops to remove noise (small dots) and fill holes
                    kernel = np.ones((7, 7), np.uint8)
                    drivable_mask = cv2.morphologyEx(drivable_mask, cv2.MORPH_CLOSE, kernel)

                    drivable_mask = region_of_interest(drivable_mask)
                    
                    #Create a green overlay for the drivable area
                    drivable_overlay = np.zeros_like(lane_image)
                    drivable_overlay[drivable_mask > 0] = [0, 255, 0] # Green color
                    

                    # --- MERGE: Combine Both Paths ---

                    # 1. Blend the green drivable area (keep this as is)
                    combo_image = cv2.addWeighted(lane_image, 1.0, drivable_overlay, 0.3, 0)

                    # 2. Prepare the Mask for the blue lines
                    # Create a grayscale mask from the line_image
                    line_gray = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)
                    # Threshold it so any pixel that is NOT black becomes 255 (white)
                    _, mask = cv2.threshold(line_gray, 1, 255, cv2.THRESH_BINARY)

                    # 3. Create an inverse mask (to cut holes in the road image)
                    mask_inv = cv2.bitwise_not(mask)

                    # 4. Black out the area of the lanes in the road image
                    img_bg = cv2.bitwise_and(combo_image, combo_image, mask=mask_inv)

                    # 5. Take only the blue lines from the line image
                    img_fg = cv2.bitwise_and(line_image, line_image, mask=mask)

                    # 6. Combine them (Add the blue lines into the "holes" we cut)
                    final_image = cv2.add(img_bg, img_fg)
                    
                    # Save and Display
                    save_path = os.path.join(out_dir, img_file)
                    cv2.imwrite(save_path, final_image)
                    
                    cv2.imshow("Lane and Drivable Area Detection", final_image)
                    if cv2.waitKey(1) == ord('q'):
                        break
                else:
                    print(f"Failed to load image: {img_file}")
            
            print("Processing complete.")
            cv2.destroyAllWindows()