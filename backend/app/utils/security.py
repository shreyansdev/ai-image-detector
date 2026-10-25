def validate_image_magic_bytes(file_bytes: bytes) -> bool:
    """
    Verifies image file signature (magic bytes) to protect against content-type sniffing.
    Supports:
    - JPEG: FF D8 FF
    - PNG: 89 50 4E 47
    - WEBP: RIFF...WEBP (RIFF at 0..4, WEBP at 8..12)
    """
    if not file_bytes or len(file_bytes) < 12:
        return False

    # Check JPEG magic bytes: FF D8 FF
    if file_bytes.startswith(b'\xff\xd8\xff'):
        return True

    # Check PNG magic bytes: 89 50 4E 47 (\x89PNG)
    if file_bytes.startswith(b'\x89PNG') or file_bytes.startswith(b'\x89\x50\x4e\x47'):
        return True

    # Check WEBP magic bytes: Starts with RIFF and contains WEBP at index 8..12
    if file_bytes.startswith(b'RIFF') and file_bytes[8:12] == b'WEBP':
        return True

    return False
