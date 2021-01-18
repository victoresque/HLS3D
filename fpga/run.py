# coding: utf-8

import struct

import cv2
import sys
import numpy as np
from time import time, sleep

sys.path.append('/home/xilinx')
import pynq
from pynq import Overlay
from pynq import allocate

WIDTH, HEIGHT = 640, 480


def hdmi_iic_init(ipIIC):
    ipIIC.send(0x39, bytes([0x41, 0x50]), 2)  # power down

    ipIIC.send(0x39, bytes([0x15, 0x01]), 2)  # 16,20,24b YCbCr 4:2:2 separate sync
    ipIIC.send(0x39, bytes([0x16, 0x35]), 2)  # 4:4:4, 8bit, style 2, YCbCr
    ipIIC.send(0x39, bytes([0x17, 0x00]), 2)  # 4:3

    ipIIC.send(0x39, bytes([0x40, 0x80]), 2)  # GC packet enable
    ipIIC.send(0x39, bytes([0x48, 0x08]), 2)  # bit right justified
    ipIIC.send(0x39, bytes([0x49, 0xa8]), 2)  # bit trimming: truncate

    # color space conversion
    ipIIC.send(0x39, bytes([0x18, 0xAC]), 2)
    ipIIC.send(0x39, bytes([0x19, 0x53]), 2)
    ipIIC.send(0x39, bytes([0x1A, 0x08]), 2)
    ipIIC.send(0x39, bytes([0x1B, 0x00]), 2)
    ipIIC.send(0x39, bytes([0x1C, 0x00]), 2)
    ipIIC.send(0x39, bytes([0x1D, 0x00]), 2)
    ipIIC.send(0x39, bytes([0x1E, 0x19]), 2)
    ipIIC.send(0x39, bytes([0x1F, 0xD6]), 2)
    ipIIC.send(0x39, bytes([0x20, 0x1C]), 2)
    ipIIC.send(0x39, bytes([0x21, 0x56]), 2)
    ipIIC.send(0x39, bytes([0x22, 0x08]), 2)
    ipIIC.send(0x39, bytes([0x23, 0x00]), 2)
    ipIIC.send(0x39, bytes([0x24, 0x1E]), 2)
    ipIIC.send(0x39, bytes([0x25, 0x88]), 2)
    ipIIC.send(0x39, bytes([0x26, 0x02]), 2)
    ipIIC.send(0x39, bytes([0x27, 0x91]), 2)
    ipIIC.send(0x39, bytes([0x28, 0x1F]), 2)
    ipIIC.send(0x39, bytes([0x29, 0xFF]), 2)
    ipIIC.send(0x39, bytes([0x2A, 0x08]), 2)
    ipIIC.send(0x39, bytes([0x2B, 0x00]), 2)
    ipIIC.send(0x39, bytes([0x2C, 0x0E]), 2)
    ipIIC.send(0x39, bytes([0x2D, 0x85]), 2)
    ipIIC.send(0x39, bytes([0x2E, 0x18]), 2)
    ipIIC.send(0x39, bytes([0x2F, 0xBE]), 2)

    ipIIC.send(0x39, bytes([0x98, 0x03]), 2)  # fixed
    ipIIC.send(0x39, bytes([0x9a, 0xe0]), 2)  # fixed
    ipIIC.send(0x39, bytes([0x9c, 0x30]), 2)  # fixed
    ipIIC.send(0x39, bytes([0x9d, 0x61]), 2)  # fixed

    ipIIC.send(0x39, bytes([0xa2, 0xa4]), 2)  # fixed
    ipIIC.send(0x39, bytes([0xa3, 0xa4]), 2)  # fixed

    ipIIC.send(0x39, bytes([0xaf, 0x06]), 2)  # HDMI mode

    # ipIIC.send(0x39, bytes([0xd7, 0x16]), 2)
    # ipIIC.send(0x39, bytes([0xd8, 0x02]), 2)
    # ipIIC.send(0x39, bytes([0xd9, 0xc0]), 2)
    # ipIIC.send(0x39, bytes([0xda, 0x10]), 2)
    # ipIIC.send(0x39, bytes([0xdb, 0x05]), 2)

    ipIIC.send(0x39, bytes([0xe0, 0xd0]), 2)  # fixed
    ipIIC.send(0x39, bytes([0xf9, 0x00]), 2)  # fixed
    ipIIC.send(0x39, bytes([0xd6, 0xc0]), 2)

    ipIIC.send(0x39, bytes([0x41, 0x10]), 2)  # power up


def hdmi_init(ipHDMI):
    H_ACTIVE_TIME = WIDTH
    H_BLANKING_TIME = 153
    H_SYNC_OFFSET = 24
    H_SYNC_WIDTH_PULSE = 24
    V_ACTIVE_TIME = HEIGHT
    V_BLANKING_TIME = 50
    V_SYNC_OFFSET = 5
    V_SYNC_WIDTH_PULSE = 8

    HCNT = H_ACTIVE_TIME + H_BLANKING_TIME
    HBP = H_BLANKING_TIME - H_SYNC_OFFSET - H_SYNC_WIDTH_PULSE
    HDMIN = H_SYNC_WIDTH_PULSE + HBP
    HDMAX = HDMIN + H_ACTIVE_TIME

    VCNT = V_ACTIVE_TIME + V_BLANKING_TIME
    VBP = V_BLANKING_TIME - V_SYNC_OFFSET - V_SYNC_WIDTH_PULSE
    VDMIN = V_SYNC_WIDTH_PULSE + VBP
    VDMAX = VDMIN + V_ACTIVE_TIME

    ipHDMI.write(0x0040, 0x0)
    ipHDMI.write(0x0048, 0x0)
    ipHDMI.write(0x0400, (H_ACTIVE_TIME<<16)+HCNT)
    ipHDMI.write(0x0404, H_SYNC_WIDTH_PULSE)
    ipHDMI.write(0x0408, (HDMAX<<16)+HDMIN)
    ipHDMI.write(0x0440, (V_ACTIVE_TIME<<16)+VCNT)
    ipHDMI.write(0x0444, V_SYNC_WIDTH_PULSE)
    ipHDMI.write(0x0448, (VDMAX<<16)+VDMIN)
    ipHDMI.write(0x004c, (255<<16)+(255<<8)+0)


def vtpg_init(ipVTPG):
    ipVTPG.write(0x0040, 0x0)
    ipVTPG.write(0x0010, HEIGHT)
    ipVTPG.write(0x0018, WIDTH)
    ipVTPG.write(0x0098, 0x1)
    ipVTPG.write(0x0020, 0x0)


def vrfb_init(ipVRFB):
    ipVRFB.write(0x0010, WIDTH)
    ipVRFB.write(0x0018, HEIGHT)
    ipVRFB.write(0x0020, 2560)
    ipVRFB.write(0x0028, 10)


def float_to_bytes(x):
    return struct.pack('f', x)

if __name__ == "__main__":
    print("Start of \"" + sys.argv[0] + "\"")

    ol = Overlay("/home/xilinx/final/final.bit")
    ipIIC = ol.axi_iic_0
    ipVTPG = ol.v_tpg_0
    ipHDMI = ol.axi_hdmi_tx_0
    ipVRFB = ol.v_frmbuf_rd_0
    pynq.ps.Clocks.fclk1_mhz = 38.5

    hdmi_iic_init(ipIIC)
    hdmi_init(ipHDMI)
    vtpg_init(ipVTPG)
    vrfb_init(ipVRFB)

    ipIDMA = ol.axi_dma_0
    ipODMA = ol.axi_dma_1

    scene = []
    with open('./scene.txt') as f:
        for line in f:
            scene.append(float(line))
    scene = np.array(scene)
    print(len(scene))

    ipRender = ol.render_0
    ipRender.write(0x0018, 1520)
    ipRender.write(0x0080, float_to_bytes(1.0))
    ipRender.write(0x0088, 0)

    ipRender.write(0x0040, float_to_bytes(1.0))
    ipRender.write(0x0044, float_to_bytes(0.0))
    ipRender.write(0x0048, float_to_bytes(0.0))
    ipRender.write(0x004C, float_to_bytes(0.0))
    ipRender.write(0x0050, float_to_bytes(0.0))
    ipRender.write(0x0054, float_to_bytes(1.0))
    ipRender.write(0x0058, float_to_bytes(0.0))
    ipRender.write(0x005C, float_to_bytes(0.0))
    ipRender.write(0x0060, float_to_bytes(0.0))
    ipRender.write(0x0064, float_to_bytes(0.0))
    ipRender.write(0x0068, float_to_bytes(1.0))
    ipRender.write(0x006C, float_to_bytes(0.0))

    ipRender.write(0x0090, float_to_bytes(0.81))
    ipRender.write(0x0094, float_to_bytes(0.55))
    ipRender.write(0x0098, float_to_bytes(0.21))

    ipRender.write(0x00A0, float_to_bytes(514.22))
    ipRender.write(0x00A4, float_to_bytes(514.22))
    ipRender.write(0x00A8, float_to_bytes(-1.0))

    ipRender.write(0x00B0, float_to_bytes(320.0))
    ipRender.write(0x00B4, float_to_bytes(240.0))
    ipRender.write(0x00B8, float_to_bytes(0.0))

    input_buffer = allocate(shape=(1520*24,), dtype=np.float32)
    output_buffer = allocate(shape=(480*640,), dtype=np.uint32)

    input_buffer[:] = scene

    t0 = time()
    for i in range(8):
        fh0 = 60 * i
        fh1 = 60 * (i + 1)
        ipRender.write(0x00C0, fh0)
        ipRender.write(0x00C4, fh1)

        ipRender.write(0x0010, 0)
        ipRender.write(0x0000, 1)

        while True:
            if (ipRender.read(0x0000) & 0x4) != 0x0:
                break

        ipRender.write(0x0010, 2)
        ipRender.write(0x0000, 1)

        ipIDMA.sendchannel.transfer(input_buffer)
        ipIDMA.sendchannel.wait()

        while True:
            if (ipRender.read(0x0000) & 0x4) != 0x0:
                break

        ipRender.write(0x0010, 3)
        ipRender.write(0x0000, 1)

        ipODMA.recvchannel.transfer(output_buffer[fh0*640:fh1*640])
        ipODMA.recvchannel.wait()

        while True:
            if (ipRender.read(0x0000) & 0x4) != 0x0:
                break
    print('time:', time() - t0)

    output_buffer = np.reshape(output_buffer, (480, 640))

    image = np.zeros((480, 640, 3)).astype(np.uint8)
    image[:, :, 0] = output_buffer % 256
    image[:, :, 1] = (output_buffer // 256) % 256
    image[:, :, 2] = (output_buffer // 256 // 256) % 256

    cv2.imwrite('test.jpg', image)

    ipVTPG.write(0x0000, 0x81)

    ipVRFB.write(0x0030, output_buffer.device_address)
    ipVRFB.write(0x0000, 0x81)

    ipHDMI.write(0x0048, 0x1)
    ipHDMI.write(0x0040, 0x1)

    input()

    ipVTPG.write(0x0000, 0x00)
    ipVRFB.write(0x0000, 0x00)

    # AXILiteS
    # 0x00 : Control signals
    #        bit 0  - ap_start (Read/Write/COH)
    #        bit 1  - ap_done (Read/COR)
    #        bit 2  - ap_idle (Read)
    #        bit 3  - ap_ready (Read)
    #        bit 7  - auto_restart (Read/Write)
    #        others - reserved
    # 0x10 : Data signal of mode
    #        bit 31~0 - mode[31:0] (Read/Write)
    # 0x14 : reserved
    # 0x18 : Data signal of num_faces
    #        bit 31~0 - num_faces[31:0] (Read/Write)
    # 0x1c : reserved
    # 0x80 : Data signal of obj_scale
    #        bit 31~0 - obj_scale[31:0] (Read/Write)
    # 0x84 : reserved
    # 0x88 : Data signal of texture_id
    #        bit 31~0 - texture_id[31:0] (Read/Write)
    # 0x8c : reserved
    # 0x40 ~
    # 0x7f : Memory 'transform' (12 * 32b)
    #        Word n : bit [31:0] - transform[n]
    # 0x90 ~
    # 0x9f : Memory 'lnorm' (3 * 32b)
    #        Word n : bit [31:0] - lnorm[n]
    # 0xa0 ~
    # 0xaf : Memory 'cam_scale' (3 * 32b)
    #        Word n : bit [31:0] - cam_scale[n]
    # 0xb0 ~
    # 0xbf : Memory 'cam_offset' (3 * 32b)
    #        Word n : bit [31:0] - cam_offset[n]
    # 0xc0 ~
    # 0xc7 : Memory 'frameh' (2 * 32b)
    #        Word n : bit [31:0] - frameh[n]
    # (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)


