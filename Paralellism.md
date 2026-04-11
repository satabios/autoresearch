Example (TP vs PP)
NVLink bandwidth:    ~600 GB/s  (H100 NVLink)
PCIe bandwidth:      ~32 GB/s
Ethernet (multi-node): ~25 Gb/s = ~3 GB/s

Without TP (1 GPU):
┌─────────────────────┐
│  GPU 0              │
│  W: [4096 x 16384]  │  ← entire matrix lives here
│  needs 256MB VRAM   │
└─────────────────────┘

With TP (4 GPUs, column-split):
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ GPU 0        │  │ GPU 1        │  │ GPU 2        │  │ GPU 3        │
│ W: [4096x4096│  │ W: [4096x4096│  │ W: [4096x4096│  │ W: [4096x4096│
│ 64MB VRAM   │  │ 64MB VRAM   │  │ 64MB VRAM   │  │ 64MB VRAM   │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
        │                 │                │                │
        └─────────────────┴────────────────┴────────────────┘
                              all-reduce (fast on NVLink)
                              assemble final output


Time →
GPU 0: [F0][F0][ idle ][ idle ][B0][B0]
GPU 1: [idle][F1][F1  ][ idle ][idle][B1]
GPU 2: [idle][idle][F2][F2   ][idle][idle]

      ↑ wasted time doing nothing
