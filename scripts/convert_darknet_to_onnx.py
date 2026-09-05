import struct, numpy as np, onnx, os
from onnx import helper, TensorProto, numpy_helper
cfg_path = "/home/torchi/Desktop/SITR/model/pordav4x3.cfg"
weights_path = "/home/torchi/Desktop/SITR/model/porda-19200-lr-0005-909.weights"
onnx_path = "/home/torchi/Desktop/SITR/model/porda.onnx"

def parse_cfg(p):
    blocks=[]
    cur=None
    with open(p) as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('['):
                cur={'type':line[1:-1].strip()}
                blocks.append(cur)
            else:
                k,v=line.split('=',1)
                cur[k.strip()]=v.strip()
    return blocks

blocks=parse_cfg(cfg_path)
net=blocks[0]
layers=blocks[1:]
net_w=int(net['width']); net_h=int(net['height']); net_c=int(net['channels'])

# re-simulate to get channel/h/w for each layer (needed for in_c)
channels_list=[]; h_list=[]; w_list=[]
def get_ch(idx):
    if idx==-1: return net_c
    return channels_list[idx]
def get_h(idx):
    if idx==-1: return net_h
    return h_list[idx]
def get_w(idx):
    if idx==-1: return net_w
    return w_list[idx]

for idx, block in enumerate(layers):
    t=block['type']
    if t=='convolutional':
        f=int(block['filters']); sz=int(block['size']); stride=int(block['stride']); pad=int(block.get('pad','0'))
        padding=sz//2 if pad else 0
        if idx==0:
            in_c=net_c; ph=net_h; pw=net_w
        else:
            in_c=channels_list[idx-1]; ph=h_list[idx-1]; pw=w_list[idx-1]
        oh=(ph+2*padding - sz)//stride +1
        ow=(pw+2*padding - sz)//stride +1
        channels_list.append(f); h_list.append(oh); w_list.append(ow)
    elif t=='route':
        layers_str=block['layers']
        groups=int(block.get('groups','1')); gid=int(block.get('group_id','0'))
        vals=[int(x.strip()) for x in layers_str.split(',')]
        resolved=[]
        for v in vals:
            if v<0: resolved.append(idx+v)
            else: resolved.append(v)
        if len(resolved)==1:
            src=resolved[0]
            if groups>1:
                src_ch=get_ch(src)
                ch=src_ch//groups
                channels_list.append(ch); h_list.append(get_h(src)); w_list.append(get_w(src))
            else:
                channels_list.append(get_ch(src)); h_list.append(get_h(src)); w_list.append(get_w(src))
        else:
            ch=sum(get_ch(r) for r in resolved)
            channels_list.append(ch); h_list.append(get_h(resolved[0])); w_list.append(get_w(resolved[0]))
    elif t=='maxpool':
        size=int(block['size']); stride=int(block['stride'])
        ch=channels_list[idx-1]; ph=h_list[idx-1]; pw=w_list[idx-1]
        oh=(ph-size)//stride+1
        ow=(pw-size)//stride+1
        channels_list.append(ch); h_list.append(oh); w_list.append(ow)
    elif t=='upsample':
        stride=int(block['stride'])
        ch=channels_list[idx-1]; ph=h_list[idx-1]; pw=w_list[idx-1]
        channels_list.append(ch); h_list.append(ph*stride); w_list.append(pw*stride)
    elif t=='yolo':
        channels_list.append(channels_list[idx-1]); h_list.append(h_list[idx-1]); w_list.append(w_list[idx-1])
    else:
        channels_list.append(channels_list[idx-1]); h_list.append(h_list[idx-1]); w_list.append(w_list[idx-1])

print("Sim done")
for i, l in enumerate(layers):
    print(i, l['type'], channels_list[i], h_list[i], w_list[i])

# Load weights
with open(weights_path,'rb') as f:
    major,minor,rev=struct.unpack('<3i', f.read(12))
    if major*10+minor>=2:
        seen=struct.unpack('<q', f.read(8))[0]
    else:
        seen=struct.unpack('<i', f.read(4))[0]
    weights=np.frombuffer(f.read(), dtype=np.float32)
ptr=0

nodes=[]
initializers=[]
value_infos=[]

# Input
input_name="input"
input_tensor = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 3, net_h, net_w])
# We'll keep list of tensor names per layer idx
tensor_names=[] # per layer idx final output name
# For input, idx -1 name is input_name
# For each layer idx, we produce output name

prev_name=input_name

# helper to add initializer
def add_initializer(name, arr):
    init=numpy_helper.from_array(arr.astype(np.float32), name)
    initializers.append(init)
    return name

cnt_conv=0
for idx, block in enumerate(layers):
    t=block['type']
    if t=='convolutional':
        filters=int(block['filters'])
        size=int(block['size'])
        stride=int(block['stride'])
        pad=int(block.get('pad','0'))
        activation=block.get('activation','linear')
        bn=int(block.get('batch_normalize','0'))
        # determine in channels
        if idx==0:
            in_c=net_c
        else:
            in_c=channels_list[idx-1]
        # padding
        padding=size//2 if pad else 0
        pads=[padding,padding,padding,padding]
        # weight reading
        # BN: 4*filters floats first
        # darknet order: bn bias (beta), bn weight (gamma), bn mean, bn var
        # Actually need confirm order: Alexey's code stores bn biases first, then weights, then mean, var?
        # We'll follow typical: bias, weight, mean, var each filters
        # Then conv weight
        if bn:
            bn_bias=weights[ptr:ptr+filters]; ptr+=filters
            bn_weight=weights[ptr:ptr+filters]; ptr+=filters
            bn_mean=weights[ptr:ptr+filters]; ptr+=filters
            bn_var=weights[ptr:ptr+filters]; ptr+=filters
            conv_weight=weights[ptr:ptr+filters*in_c*size*size]; ptr+=filters*in_c*size*size
            conv_weight=np.reshape(conv_weight, (filters, in_c, size, size))
            # create initializers
            w_name=f"conv{idx}_weight"
            add_initializer(w_name, conv_weight)
            # BN initials
            scale_name=f"bn{idx}_scale"
            bias_name=f"bn{idx}_bias"
            mean_name=f"bn{idx}_mean"
            var_name=f"bn{idx}_var"
            add_initializer(scale_name, bn_weight)
            add_initializer(bias_name, bn_bias)
            add_initializer(mean_name, bn_mean)
            add_initializer(var_name, bn_var)
            # Conv node
            conv_out=f"conv{idx}_out"
            conv_node=helper.make_node("Conv", inputs=[prev_name, w_name], outputs=[conv_out],
                kernel_shape=[size,size], strides=[stride,stride], pads=pads)
            nodes.append(conv_node)
            # BN node
            bn_out=f"bn{idx}_out"
            bn_node=helper.make_node("BatchNormalization", inputs=[conv_out, scale_name, bias_name, mean_name, var_name],
                outputs=[bn_out], epsilon=1e-5, momentum=0.9)
            nodes.append(bn_node)
            inter=bn_out
        else:
            # bias then weight
            conv_bias=weights[ptr:ptr+filters]; ptr+=filters
            conv_weight=weights[ptr:ptr+filters*in_c*size*size]; ptr+=filters*in_c*size*size
            conv_weight=np.reshape(conv_weight, (filters, in_c, size, size))
            w_name=f"conv{idx}_weight"
            b_name=f"conv{idx}_bias"
            add_initializer(w_name, conv_weight)
            add_initializer(b_name, conv_bias)
            conv_out=f"conv{idx}_out"
            conv_node=helper.make_node("Conv", inputs=[prev_name, w_name, b_name], outputs=[conv_out],
                kernel_shape=[size,size], strides=[stride,stride], pads=pads)
            nodes.append(conv_node)
            inter=conv_out
        # activation
        if activation=='leaky':
            act_out=f"act{idx}_out"
            act_node=helper.make_node("LeakyRelu", inputs=[inter], outputs=[act_out], alpha=0.1)
            nodes.append(act_node)
            out_name=act_out
        else: # linear
            out_name=inter
        # store
        # ensure tensor_names length
        # we need mapping for layer idx to output name; for convolutional we have out_name
        # For prev_name handling, next layer's prev is this out_name
        # But need to fill tensor_names list for all preceding idx already? We fill incremental
        # We'll maintain list tensor_names where index corresponds to layer idx output name
        # For non-conv layers already handled? For conv we set
        if len(tensor_names)<=idx:
            # extend
            while len(tensor_names)<=idx:
                tensor_names.append(None)
        tensor_names[idx]=out_name
        prev_name=out_name
        cnt_conv+=1
    elif t=='route':
        layers_str=block['layers']
        groups=int(block.get('groups','1')); gid=int(block.get('group_id','0'))
        vals=[int(x.strip()) for x in layers_str.split(',')]
        resolved=[]
        for v in vals:
            if v<0: resolved.append(idx+v)
            else: resolved.append(v)
        if len(resolved)==1:
            src=resolved[0]
            src_name = input_name if src==-1 else tensor_names[src]
            if groups>1:
                # split via Split
                src_ch=get_ch(src)
                # Split into groups equal parts
                split_out1=f"route{idx}_split0"
                split_out2=f"route{idx}_split1"
                # Need to decide group sizes; assume 2 groups equal
                # ONNX Split with split attribute
                if groups==2:
                    split_node=helper.make_node("Split", inputs=[src_name], outputs=[split_out1, split_out2], axis=1, split=[src_ch//2, src_ch//2])
                    nodes.append(split_node)
                    out_name = split_out2 if gid==1 else split_out1
                else:
                    raise NotImplementedError("groups !=2")
            else:
                out_name=src_name # alias, but need identity? We can just reuse name without node
                # However to make distinct, we keep same name? For graph continuity, prev will be src. But need tensor_names[idx] to point to src_name.
                # No extra node needed.
                pass
        else:
            # concat
            inputs=[]
            for r in resolved:
                n = input_name if r==-1 else tensor_names[r]
                inputs.append(n)
            out_name=f"route{idx}_out"
            concat_node=helper.make_node("Concat", inputs=inputs, outputs=[out_name], axis=1)
            nodes.append(concat_node)
        if len(tensor_names)<=idx:
            while len(tensor_names)<=idx:
                tensor_names.append(None)
        # If alias case without node, tensor_names[idx] = src_name (still reference)
        # For split case, out_name is split output
        # For concat case, out_name defined
        # So ensure we set correctly
        # Need to handle alias without defining out_name variable (we set out_name=src_name)
        # if len(resolved)==1 and groups==1 -> out_name = src_name already
        # Let's set:
        if 'out_name' not in locals():
            out_name = tensor_names[idx] if tensor_names[idx] else src_name
        else:
            # out_name already set
            pass
        tensor_names[idx]=out_name
        prev_name=out_name
        # clean out_name variable for next iteration? keep but will be overwritten
        # Need to delete to avoid carry
        if 'out_name' in locals():
            # keep for next conv use but will be overwritten
            pass
        # remove variable to avoid confusion
        try:
            del out_name
        except:
            pass
    elif t=='maxpool':
        size=int(block['size']); stride=int(block['stride'])
        out_name=f"maxpool{idx}_out"
        m_node=helper.make_node("MaxPool", inputs=[prev_name], outputs=[out_name], kernel_shape=[size,size], strides=[stride,stride], pads=[0,0,0,0])
        nodes.append(m_node)
        if len(tensor_names)<=idx:
            while len(tensor_names)<=idx:
                tensor_names.append(None)
        tensor_names[idx]=out_name
        prev_name=out_name
    elif t=='upsample':
        stride=int(block['stride'])
        out_name=f"upsample{idx}_out"
        scales_name=f"upsample{idx}_scales"
        roi_name=f"upsample{idx}_roi"
        scales_arr=np.array([1.0,1.0,float(stride),float(stride)], dtype=np.float32)
        # roi empty 0 elements - required for Resize opset11 but can be empty; we store as empty initializer
        roi_arr=np.array([], dtype=np.float32)
        add_initializer(roi_name, roi_arr)
        add_initializer(scales_name, scales_arr)
        node=helper.make_node("Resize", inputs=[prev_name, roi_name, scales_name], outputs=[out_name], mode="nearest", coordinate_transformation_mode="asymmetric")
        nodes.append(node)
        if len(tensor_names)<=idx:
            while len(tensor_names)<=idx:
                tensor_names.append(None)
        tensor_names[idx]=out_name
        prev_name=out_name
    elif t=='yolo':
        # yolo layer: no op, just passthrough reference? Actually tensor_names[idx] already set to previous? But need to keep prev unchanged?
        # Yolo does not produce new tensor; its output is previous conv but not used as input for next? However next route -4 references conv27 not yolo, so fine.
        # We'll set tensor_names[idx] = prev_name (which is conv output)
        if len(tensor_names)<=idx:
            while len(tensor_names)<=idx:
                tensor_names.append(None)
        tensor_names[idx]=prev_name
        # prev remains same (yolo doesn't change)
        # no node
    else:
        raise ValueError(f"unknown {t}")

print("ptr", ptr, "total", weights.size, "diff", weights.size-ptr)
assert ptr==weights.size, "weight consumption mismatch"

# Define outputs: conv29 (idx 29) and conv36 (idx 36) => tensor_names[29] and tensor_names[36]
out1_name=tensor_names[29]
out2_name=tensor_names[36]
print("outputs", out1_name, out2_name, "shapes", channels_list[29], h_list[29], w_list[29], channels_list[36], h_list[36], w_list[36])
# Value infos
output1 = helper.make_tensor_value_info(out1_name, TensorProto.FLOAT, [1, channels_list[29], h_list[29], w_list[29]])
output2 = helper.make_tensor_value_info(out2_name, TensorProto.FLOAT, [1, channels_list[36], h_list[36], w_list[36]])

graph = helper.make_graph(nodes, "Porda", [input_tensor], [output1, output2], initializer=initializers)
# Use opset 11 for Resize
opset = helper.make_operatorsetid("", 11)
model = helper.make_model(graph, opset_imports=[opset], producer_name="porda-darknet2onnx")
# Check
onnx.checker.check_model(model)
print("model checked ok")
# Save
onnx.save(model, onnx_path)
print(f"saved to {onnx_path} size {os.path.getsize(onnx_path)}")
# Print info
print("inputs:", model.graph.input[0])
print("outputs:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in model.graph.output])
# Save helper text
try:
    print(onnx.helper.printable_graph(model.graph)[:5000])
except: pass
