from functions.train_meg_things_pipeline import *
import shutil
from scipy.interpolate import interp1d

def smoothen_rdm(rdm_array, window = 5):
    # Keep the first timepoint's RDM as is
    first_rdm = rdm_array[0:1]  # Shape: (1, 118, 118)

    # Process the remaining timepoints
    remaining_rdms = rdm_array[1:]  # Shape: (1100, 118, 118)

    # Calculate the number of groups (each with 5 timepoints)
    num_groups = remaining_rdms.shape[0] // window

    # Reshape the remaining RDMs to group every 5 timepoints together
    reshaped_array = remaining_rdms[:num_groups * window].reshape(num_groups, window, 118, 118)

    # Average each group along the second axis
    averaged_rdms = reshaped_array.mean(axis=1)  # Shape: (num_groups, 118, 118)

    # Concatenate the first RDM with the averaged RDMs
    result = np.concatenate([first_rdm, averaged_rdms], axis=0)

    return result

def smoothen_weighting_matrix(weighting_matrix, window=5):
    # input shape: (n_timepoints, 24) output should be (n_timepoints//window + 1, 24)

    # Keep the first timepoint's weighting matrix as is
    first_weighting_matrix = weighting_matrix[0:1]  # Shape: (1, 24)

    # Process the remaining timepoints
    remaining_weighting_matrix = weighting_matrix[1:]  # Shape: (1100, 24)

    # Calculate the number of groups (each with 5 timepoints)
    num_groups = remaining_weighting_matrix.shape[0] // window

    # Reshape the remaining weighting matrices to group every 5 timepoints together
    reshaped_array = remaining_weighting_matrix[:num_groups * window].reshape(num_groups, window, 24)

    # Average each group along the second axis

    averaged_weighting_matrix = reshaped_array.mean(axis=1)  # Shape: (num_groups, 24)

    # Concatenate the first weighting matrix with the averaged weighting matrices

    result = np.concatenate([first_weighting_matrix, averaged_weighting_matrix], axis=0)

    return result







class DynamicDataset_p(Dataset):
    def __init__(self, csv_file, img_dir, rdm_dir, p_id, smoothen_window = 5):
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.52997664, 0.48070561, 0.41943838],
                                 std=[0.27608301, 0.26593025, 0.28238822])
        ])
        
        # Read the CSV file and store the image names and indices
        self.annotations = pd.read_csv(csv_file, index_col=0)
        self.image_names = self.annotations.iloc[:, 0].tolist()
        self.image_indices = self.annotations.index.tolist()
        
        # Load the full RDM matrix
        self.rdms = load_rdm(rdm_dir)
        self.rdms = self.rdms[p_id, :, :, :]
        if smoothen_window > 1:
            self.rdms = smoothen_rdm(self.rdms, window=smoothen_window)
        
        # Create a mapping from image names to indices for efficient lookup
        self.image_name_to_index = {name: idx for name, idx in zip(self.image_names, self.image_indices)}

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Get the index of the image_name using the precomputed mapping
        image_name_index = self.image_name_to_index[image_name]

        return image_name, image, image_name_index

def compute_rdm_generalization(rdm, zero_ms_position): 
    def flatten_rdm(rdm):
        triu_indices = np.triu_indices(rdm.shape[-1], k=1)
        return rdm[..., triu_indices[0], triu_indices[1]]
    
    def compute_time_rsm(rdm):
        rdm_flat = flatten_rdm(rdm)
        time_rsm = np.corrcoef(rdm_flat)
        # time_rsm = 1 - squareform(pdist(rdm_flat, metric='euclidean'))
        return time_rsm
    

    time_rsm = compute_time_rsm(rdm[:, :, :])
    generalization = np.mean(time_rsm, axis=1)
    generalization_min_maxed = (generalization - generalization.min()) / (generalization.max() - generalization.min())

    generalization_min_maxed[:zero_ms_position] = 0

    # generalization_min_maxed = gaussian_filter1d(generalization_min_maxed, sigma=3)

    
    return generalization_min_maxed

def get_richness(rdms, zero_ms_position):


    richness = np.mean(rdms, axis=(1, 2))

    richness_min_maxed = (richness - richness.min()) / (richness.max() - richness.min())

    richness_min_maxed[:zero_ms_position] = 0

    # richness_min_maxed = gaussian_filter1d(richness_min_maxed, sigma=3)

    return richness_min_maxed


def train_model(model, train_loader, test_loader, device, criterion, p_weight, m_weight, g_weight, optimizer_0, optimizer_1, epochs, fw_tuning_epochs, rdms, sample_timepoints, ms_start, ms_end, ms_step, window_size, early_stopping_patience_0, early_stopping_patience_1, checkpoint_path='clip_hba_model_cv.pth'):

    model.train()
    best_test_loss = p_weight + m_weight + g_weight
    best_pearson_loss = 999
    epochs_no_improve = 0
    optimizer = optimizer_0

    train_rdms = rdms[:, :, :]
    test_rdms = rdms[:, :, :]

    # # Convert train_loader to list so we can shuffle it later
    # train_data = list(train_loader)


    # Initial evaluation
    print("\n--- Initial Evaluation Starting ---")
    t, m, p, g = evaluate_model(model, test_loader, device, criterion, test_rdms, sample_timepoints, ms_start, ms_end, ms_step, window_size, optimizer)
    best_pearson_loss = p
    criterion = PearsonMSELongLoss(initial_mse_loss=m, initial_pearson_loss=p, initial_generalization_loss=g, p_weight=p_weight, m_weight=m_weight, g_weight=g_weight)
    print(f"Initial Validation Loss: T={best_test_loss:.4f}, M={m:.4f}, P={p:.4f}, G={g:.4f}")
    print("--- Initial Evaluation Complete ---\n")


    print("--- Training Starting ---")
    for epoch in range(epochs):
        
        if fw_tuning_epochs is not None:

            if epoch > fw_tuning_epochs - 1:
                optimizer = optimizer_1
            else:
                optimizer = optimizer_0
                epochs_no_improve = 0

            if epoch == fw_tuning_epochs:
                print("\n\n*********************************")
                print(f"ViT Starts training at epoch {epoch+1}")
                print("*********************************\n\n")

                # load the latest checkpoint
                model.load_state_dict(torch.load(checkpoint_path))
        
        if optimizer == optimizer_0:
            patience = early_stopping_patience_0
        else:
            patience = early_stopping_patience_1


        
        total_loss = 0.0 
        total_iterations = len(train_loader)

        # # Shuffle train data at the start of each epoch
        # random.shuffle(train_data)


        progress_bar = tqdm(total=total_iterations, desc=f"Epoch {epoch+1}/{epochs}")


        for batch_idx, (image_name, images, indices) in enumerate(train_loader):

            # # Shuffle the images and indices within the batch
            # perm = torch.randperm(images.size(0))  # Generate a random permutation
            # images = images[perm]
            # indices = [indices[i] for i in perm.tolist()]

            # print(f"Indices: {indices}")

            images = images.to(device)

            optimizer.zero_grad()

            pred_emb_3d, pred_rdm_3d, _ = model(images)

            target_rdm_3d = train_rdms[:, np.ix_(indices, indices)[0], np.ix_(indices, indices)[1]]
            target_rdm_3d = torch.tensor(target_rdm_3d, dtype=torch.float).to(device)

            loss, mse_loss, pearson_loss, g_loss = criterion(pred_rdm_3d, target_rdm_3d)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({'Loss': loss.item(), 'M': mse_loss.item(), 'P': pearson_loss.item(), 'G': g_loss.item()})
            progress_bar.update(1)
        progress_bar.close()
                

                
        avg_train_loss = total_loss / total_iterations  # Average loss for the epoch
        progress_bar.close()

        # Evaluate after every epoch
        avg_test_loss, avg_mse_loss, avg_pearson_loss, avg_g_loss = evaluate_model(model, test_loader, device, criterion, test_rdms, sample_timepoints, ms_start, ms_end, ms_step, window_size, optimizer)
        print(f"Epoch {epoch+1}: Training Loss: T={avg_train_loss:.4f}, Validation Loss: T={avg_test_loss:.4f}, M={avg_mse_loss:.4f}, P={avg_pearson_loss:.4f}, G={avg_g_loss:.4f}")
        
        # Check for early stopping and saving checkpoint
        if avg_test_loss < best_test_loss and avg_pearson_loss < best_pearson_loss:
        # if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_pearson_loss = avg_pearson_loss
            epochs_no_improve = 0
            # Save the model checkpoint
            torch.save(model.state_dict(), checkpoint_path)
            print("\n\n-----------------------------------")
            print(f"Checkpoint saved for epoch {epoch+1}")
            print("-----------------------------------\n\n")
            # break # test break
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            
            if optimizer == optimizer_0:
                print("\n\n*********************************")
                print(f"ViT Starts training at epoch {epoch+1}")
                print("*********************************\n\n")
                optimizer = optimizer_1
                epochs_no_improve = 0
                model.load_state_dict(torch.load(checkpoint_path))
                fw_tuning_epochs = None
            else:
                print("\n\n*********************************")
                print(f"Early stopping triggered at epoch {epoch+1}")
                print("*********************************\n\n")
                break


    print("--- Training Complete ---\n")



def evaluate_model(model, data_loader, device, criterion, test_rdms, sample_timepoints, ms_start, ms_end, ms_step, window_size, optimizer):
    model.eval()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_pearson_loss = 0.0
    total_g_loss = 0.0
    total_iterations = len(data_loader)
    progress_bar = tqdm(total=total_iterations, desc="Evaluating")
    
    with torch.no_grad():
        for batch_idx, (_, images, indices) in enumerate(data_loader):
            images = images.to(device)

            optimizer.zero_grad()

            pred_emb_3d, pred_rdm_3d, _ = model(images)
        

            target_rdm_3d = test_rdms[:, np.ix_(indices, indices)[0], np.ix_(indices, indices)[1]]
            target_rdm_3d = torch.tensor(target_rdm_3d, dtype=torch.float).to(device)

            loss, mse_loss, pearson_loss, g_loss = criterion(pred_rdm_3d, target_rdm_3d)

            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_pearson_loss += pearson_loss.item()
            total_g_loss += g_loss.item()

            progress_bar.set_postfix({'Loss': loss.item(), 'M': mse_loss.item(), 'P': pearson_loss.item(), 'G': g_loss.item()})
            progress_bar.update(1)

    progress_bar.close()
    avg_loss = total_loss / total_iterations
    avg_mse_loss = total_mse_loss / total_iterations
    avg_pearson_loss = total_pearson_loss / total_iterations
    avg_g_loss = total_g_loss / total_iterations

    return avg_loss, avg_mse_loss, avg_pearson_loss, avg_g_loss



def load_pretrained_weights_with_adjustments(model, pretrained_path, train_start, train_end, train_step, pre_trained_fw_matrix=False):
    """
    Custom function to load pretrained weights while handling mismatched dimensions.
    """
    model_state_dict = torch.load(pretrained_path)
    adjusted_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}

    # Define old timepoints
    old_timepoints = np.arange(-100, 1301, 5)
    new_timepoints = np.arange(train_start, train_end + 1, train_step)
    
    # Handle mismatched parameters
    for key in list(adjusted_state_dict.keys()):
        if key in ["clip_model.beta", "clip_model.noise_level", "clip_model.visual_scaler", "clip_model.weighting_matrix"]:
            # Skip these parameters as they're redefined
            del adjusted_state_dict[key]

    
    # Load the adjusted state_dict
    model.load_state_dict(adjusted_state_dict, strict=False)
    print("Loaded pretrained weights with adjustments.")


def run_individual_training(config):
    """
    Run individual MEG training with the given configuration.
    
    Args:
        config (dict): Configuration dictionary containing training parameters
    """
    seed_everything(config['random_seed'])

    # Create model save folder
    if os.path.exists(config['model_save_folder']):
        print("Model save folder already exists.")
        input(">>>>>>>>>>>>>>>>>  Press ENTER to reinitialize and continue  <<<<<<<<<<<<<<<<<")
        shutil.rmtree(config['model_save_folder'])
    os.makedirs(config['model_save_folder'])

    # Train for each participant
    for p_id in range(config['n_participants']):
        if p_id in config['skip_id']:
            print(f"Skipping Participant {p_id+1}")
            continue
        
        print("**********")
        print(f"Training for Participant {p_id+1}")
        print("**********")

        # Initialize classnames
        classnames = [x[0] for x in classnames66]
        
        # Load dataset
        dataset = DynamicDataset_p(csv_file=config['csv_file'],
                                 img_dir=config['img_dir'],
                                 rdm_dir=config['rdm_dir'],
                                 p_id=p_id,
                                 smoothen_window=config['smoothen_window'])
        rdms = dataset.rdms

        # Define curves
        sample_timepoints = list(range(config['train_start'],
                                     config['train_end']+1,
                                     config['train_step']))
        zero_ms_position = (0 - config['ms_start']) // config['ms_step'] + 1
        beta = compute_rdm_generalization(rdms, zero_ms_position)
        alpha = get_richness(rdms, zero_ms_position)
        noise_level = compute_noise_level(beta, scale=config['noise_scale'])

        # Split dataset
        train_size = int(config['train_portion'] * len(dataset))
        test_size = len(dataset) - train_size
        print(f"\nTrain size: {train_size}, Test size: {test_size}\n")
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Create data loaders
        train_loader = DataLoader(train_dataset,
                                batch_size=config['batch_size'],
                                shuffle=True)
        test_loader = DataLoader(test_dataset,
                               batch_size=config['batch_size'],
                               shuffle=False)
        
        # Set position embedding based on backbone
        pos_embedding = False if config['backbone'] == 'RN50' else True

        # Initialize model
        model = CLIPHBA(classnames=classnames,
                       weighting_matrix=None,
                       backbone_name=config['backbone'],
                       pos_embedding=pos_embedding,
                       ms_start=config['ms_start'],
                       ms_step=config['ms_step'],
                       ms_end=config['ms_end'],
                       train_start=config['train_start'],
                       train_step=config['train_step'],
                       train_end=config['train_end'],
                       train_window_size=config['train_window_size'],
                       beta=beta,
                       noise_level=noise_level,
                       visual_scaler=alpha)

        # Verify model dimensions
        assert model.clip_model.weighting_matrix.shape[0] == len(sample_timepoints), \
            "Weighting matrix shape mismatch"

        # Set device
        if config['cuda'] == -1:
            device = torch.device("cuda")
            print(f"Using {torch.cuda.device_count()} GPUs")
        elif config['cuda'] == 0:
            device = torch.device("cuda:0")
            print("Using GPU 0")
        elif config['cuda'] == 1:
            device = torch.device("cuda:1")
            print("Using GPU 1")
        else:
            device = torch.device("cpu")
            print("Using CPU")

        # Apply DoRA
        apply_dora_to_ViT(model,
                         n_vision_layers=config['vision_layers'],
                         n_transformer_layers=0,
                         r=config['rank'],
                         dora_dropout=0.1,
                         seed=config['random_seed'])
        apply_dora_to_ViT(model,
                         n_vision_layers=0,
                         n_transformer_layers=config['transformer_layers'],
                         r=32,
                         dora_dropout=0.1,
                         seed=config['random_seed'])
        
        switch_dora_layers(model,
                         freeze_all=True,
                         d_state=config['dora_d_state'],
                         m_state=config['dora_m_state'])
        unfreeze_weighting_parameters(model)

        # Load pretrained text encoder
        if config['text_encoder_path']:
            print(f"Loading pretrained model: {config['text_encoder_path']}")
            model_state_dict = torch.load(config['text_encoder_path'])
            adjusted_state_dict = {key.replace("module.", ""): value 
                                 for key, value in model_state_dict.items()}
            model.load_state_dict(adjusted_state_dict, strict=False)
            
            if config['freeze_text']:
                freeze_text_encoder(model)
                print("Model text encoder frozen\n")

        if config['cuda'] == -1:
            model = DataParallel(model)
        model.to(device)

        # Initialize optimizers
        optimizer_0 = AdamW([
            {'params': [p for n, p in model.named_parameters() 
                       if n != 'clip_model.weighting_matrix'],
             'lr': config['lr_1']},
            {'params': [model.clip_model.weighting_matrix],
             'lr': config['fw_lr_1']}
        ])

        optimizer_1 = AdamW([
            {'params': [p for n, p in model.named_parameters() 
                       if n != 'clip_model.weighting_matrix'],
             'lr': config['lr_2']},
            {'params': [model.clip_model.weighting_matrix],
             'lr': config['fw_lr_2']}
        ])

        criterion = PearsonMSELongLoss(p_weight=config['p_weight'],
                                     m_weight=config['m_weight'],
                                     g_weight=config['g_weight'])

        # Print training information
        print("Updating layers:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
        print(f"Number of trainable parameters: {count_trainable_parameters(model)}\n")

        # Train model
        checkpoint_path = f"{config['model_save_folder']}/cliphba_dynamic_individual_cichy_p{p_id+1}.pth"
        train_model(model, train_loader, test_loader, device,
                   criterion, config['p_weight'], config['m_weight'], config['g_weight'],
                   optimizer_0, optimizer_1, config['epochs'],
                   config['fw_tuning_epochs'], rdms, sample_timepoints,
                   config['ms_start'], config['ms_end'], config['ms_step'],
                   config['train_window_size'],
                   config['early_stopping_patience_0'],
                   config['early_stopping_patience_1'],
                   checkpoint_path)
