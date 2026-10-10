import numpy as np
from pycocotools import mask as mask_utils


def encode_rle(binary_mask: np.ndarray) -> dict:
    """
    Encode binary mask to RLE format using pycocotools.
    
    Args:
        binary_mask: 2D binary numpy array (H, W) with values 0 or 1
        
    Returns:
        RLE dict with 'size' and 'counts' keys
    """
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    
    # Ensure fortran order for pycocotools
    binary_mask = np.asfortranarray(binary_mask)
    rle = mask_utils.encode(binary_mask)
    
    # Convert counts to string for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def decode_rle(rle: dict, height: int = 2048, width: int = 2048) -> np.ndarray:
    """
    Decode RLE to binary mask.
    
    Args:
        rle: RLE dict with 'size' and 'counts' keys
        height: Height of the mask
        width: Width of the mask
        
    Returns:
        2D binary numpy array (H, W)
    """
    if isinstance(rle['counts'], str):
        rle['counts'] = rle['counts'].encode('utf-8')
    binary_mask = mask_utils.decode(rle)
    return binary_mask.astype(np.uint8)


def polygon_to_rle(polygon: list, height: int = 2048, width: int = 2048) -> dict:
    """
    Convert COCO polygon format to RLE.
    
    Args:
        polygon: List of polygon coordinates [x0, y0, x1, y1, ...]
        height: Image height
        width: Image width
        
    Returns:
        RLE dict
    """
    if isinstance(polygon[0], list):
        # Multiple polygons (list of polygons)
        rles = mask_utils.frPyObjects(polygon, height, width)
        rle = mask_utils.merge(rles)
    else:
        # Single polygon
        rle = mask_utils.frPyObjects(polygon, height, width)
    return rle


def rle_to_binary_mask(rle: dict, height: int = 2048, width: int = 2048) -> np.ndarray:
    """
    Convert RLE to binary mask (alias for decode_rle).
    
    Args:
        rle: RLE dict
        height: Image height
        width: Image width
        
    Returns:
        Binary mask
    """
    return decode_rle(rle, height, width)


def binary_mask_to_polygon(binary_mask: np.ndarray) -> list:
    """
    Convert binary mask to COCO polygon format.
    
    Args:
        binary_mask: 2D binary numpy array
        
    Returns:
        List of polygon coordinates
    """
    import cv2
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        if contour.size >= 6:  # At least 3 points
            polygon = contour.flatten().tolist()
            polygons.append(polygon)
    
    return polygons


def filter_rle_by_area(rle: dict, min_area: int = 150) -> bool:
    """
    Check if RLE mask has area >= min_area.
    
    Args:
        rle: RLE dict
        min_area: Minimum area threshold
        
    Returns:
        True if area >= min_area
    """
    area = mask_utils.area(rle)
    return area >= min_area


def rle_iou(rle1: dict, rle2: dict) -> float:
    """
    Compute IoU between two RLE masks.
    
    Args:
        rle1: First RLE mask
        rle2: Second RLE mask
        
    Returns:
        IoU score
    """
    return mask_utils.iou([rle1], [rle2], [False])[0][0]


def merge_rles(rles: list) -> dict:
    """
    Merge multiple RLE masks into one.
    
    Args:
        rles: List of RLE dicts
        
    Returns:
        Merged RLE dict
    """
    return mask_utils.merge(rles)


def area_rle(rle: dict) -> float:
    """
    Compute area of RLE mask.
    
    Args:
        rle: RLE dict
        
    Returns:
        Area in pixels
    """
    return mask_utils.area(rle)


def to_submission_format(rle: dict, image_id: str, idx: int) -> tuple:
    """
    Convert RLE to submission format.
    
    Args:
        rle: RLE dict
        image_id: Image identifier (e.g., '010401-20160920230134Lh')
        idx: Instance index
        
    Returns:
        Tuple of (filament_id, segmentation_rle_string)
    """
    filament_id = f"{image_id}_{idx}"
    # Return raw RLE counts string without quotes
    segmentation_rle = rle['counts']
    return filament_id, segmentation_rle