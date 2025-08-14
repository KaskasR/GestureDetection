# parse_tlvs.py
import struct, os, glob, argparse
import numpy as np

MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'  # mmWave "magic word"

# Generic mmWave frame header (xWR68xx demo style)
# After magic word (8B), header is typically 40 bytes:
# uint32 version, uint32 totalLen, uint32 platform, uint32 frameNum,
# uint32 cpuCycles, uint32 numDetectedObj, uint32 numTLVs, uint32 subFrameNum
HDR_LEN = 32
TLV_HDR_LEN = 8  # uint32 type, uint32 length

def find_all(data: bytes, pat: bytes):
    i = 0
    while True:
        i = data.find(pat, i)
        if i == -1: break
        yield i
        i += 1

def parse_file(path, want_tlv_type=1050):
    with open(path, 'rb') as f:
        data = f.read()

    feats = []  # list of np.float32 arrays, one per frame (for tlv type)
    for mpos in find_all(data, MAGIC):
        # Check header bounds
        if mpos + 8 + HDR_LEN > len(data):
            continue
        hdr = data[mpos+8 : mpos+8+HDR_LEN]
        version, total_len, platform, frame_num, cpu, num_obj, num_tlvs, subframe = struct.unpack('<8I', hdr)

        # Sanity on packet length
        end = mpos + total_len
        if end > len(data) or total_len < (8 + HDR_LEN + TLV_HDR_LEN):
            continue

        # Walk TLVs
        tlv_pos = mpos + 8 + HDR_LEN
        for _ in range(num_tlvs):
            if tlv_pos + TLV_HDR_LEN > end: break
            tlv_type, tlv_len = struct.unpack('<II', data[tlv_pos:tlv_pos+TLV_HDR_LEN])
            tlv_payload_start = tlv_pos + TLV_HDR_LEN
            tlv_payload_end = tlv_pos + tlv_len
            if tlv_payload_end > end:
                break

            if tlv_type == want_tlv_type:
                payload = data[tlv_payload_start:tlv_payload_end]
                # Expect feature vector as float32s
                if len(payload) % 4 == 0 and len(payload) > 0:
                    vec = np.frombuffer(payload, dtype=np.float32)
                    feats.append(vec.copy())

            # Advance
            tlv_pos += tlv_len

    feats = np.stack(feats, axis=0) if len(feats) else np.zeros((0,8), dtype=np.float32)
    return feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infile', help='Single .bin file')
    ap.add_argument('--indir', help='Directory with .bin files')
    ap.add_argument('--tlv', type=int, default=1050, help='TLV type to extract (default 1050)')
    ap.add_argument('--out', default='training/features', help='Where to save .npy feature files')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    files = []
    if args.infile:
        files = [args.infile]
    elif args.indir:
        files = sorted(glob.glob(os.path.join(args.indir, '*.bin')))
    else:
        print('Provide --infile or --indir'); return

    for fp in files:
        feats = parse_file(fp, want_tlv_type=args.tlv)
        base = os.path.splitext(os.path.basename(fp))[0]
        outp = os.path.join(args.out, base + '.npy')
        np.save(outp, feats)
        print(f'[OK] {base}: frames={feats.shape[0]} feat_dim={(feats.shape[1] if feats.size else 0)} -> {outp}')

if __name__ == '__main__':
    main()
