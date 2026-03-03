with open('functions/train_mem_pipeline.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = (
    "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    "    model.to(device)\n"
    "\n"
    "    optimizer = torch.optim.AdamW(model.mlp_parameters(),\n"
    "                                lr=config['lr'])"
)

new = (
    "    if config['cuda'] == -1:\n"
    "        device = torch.device('cuda')\n"
    "    elif config['cuda'] == 0:\n"
    "        device = torch.device('cuda:0')\n"
    "    elif config['cuda'] == 1:\n"
    "        device = torch.device('cuda:1')\n"
    "    else:\n"
    "        device = torch.device('cpu')\n"
    "\n"
    "    # Optimizer created before DataParallel wrapping so mlp_parameters() is accessible\n"
    "    optimizer = torch.optim.AdamW(model.mlp_parameters(), lr=config['lr'])\n"
    "\n"
    "    # Use DataParallel if using all GPUs\n"
    "    if config['cuda'] == -1:\n"
    "        print(f'Using {torch.cuda.device_count()} GPUs')\n"
    "        model = DataParallel(model)\n"
    "\n"
    "    model.to(device)"
)

if old in content:
    content = content.replace(old, new)
    with open('functions/train_mem_pipeline.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Success')
else:
    print('NOT FOUND - showing context:')
    idx = content.find("device = torch.device")
    print(repr(content[idx-4:idx+200]))
