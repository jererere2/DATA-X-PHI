import struct

def read_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')

def parse_gguf_header(filename):
    with open(filename, 'rb') as f:
        magic = f.read(4)
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        kv_count = struct.unpack('<Q', f.read(8))[0]
        
        print(f"Version: {version}, Tensors: {tensor_count}, KVs: {kv_count}")
        
        for i in range(kv_count):
            key = read_string(f)
            val_type = struct.unpack('<I', f.read(4))[0]
            
            val = None
            if val_type == 0: # UINT8
                val = struct.unpack('<B', f.read(1))[0]
            elif val_type == 1: # INT8
                val = struct.unpack('<b', f.read(1))[0]
            elif val_type == 2: # UINT16
                val = struct.unpack('<H', f.read(2))[0]
            elif val_type == 3: # INT16
                val = struct.unpack('<h', f.read(2))[0]
            elif val_type == 4: # UINT32
                val = struct.unpack('<I', f.read(4))[0]
            elif val_type == 5: # INT32
                val = struct.unpack('<i', f.read(4))[0]
            elif val_type == 6: # FLOAT32
                val = struct.unpack('<f', f.read(4))[0]
            elif val_type == 7: # BOOL
                val = struct.unpack('<?', f.read(1))[0]
            elif val_type == 8: # STRING
                val = read_string(f)
            elif val_type == 9: # ARRAY
                atype = struct.unpack('<I', f.read(4))[0]
                alen = struct.unpack('<Q', f.read(8))[0]
                f.read(alen) # skip
                val = f"ARRAY of len {alen}"
            elif val_type == 10: # UINT64
                val = struct.unpack('<Q', f.read(8))[0]
            elif val_type == 11: # INT64
                val = struct.unpack('<q', f.read(8))[0]
            elif val_type == 12: # FLOAT64
                val = struct.unpack('<d', f.read(8))[0]
            else:
                print(f"Unknown type {val_type} for key {key}")
                break
                
            if "context_length" in key or "embedding_length" in key or "block_count" in key:
                print(f"{key}: {val}")
            
parse_gguf_header('/run/media/administrator/DATA-X-PHI/hermes-4_3_36b-Q4_K_M.gguf')
