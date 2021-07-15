## Summary of method 
An augmentation policy network (`Controller`) is used as an adversary which attempts to increase the training loss of
the target network (`e.g. RESNET model`) through adversarial learning through adversarial learning. The target network
is trained by a larger batch conjured from multiple augmented instances of the smaller batch to promote invariant
learning. The losses of different augmented policies are applied to on the same data are used to train the augmentation
policy network by Reinforcement learning.

<b> Possible issues with implementation </b>
1. Controller class `forward` method is called only once every epoch


<b> Possible issues with **new** pytorch-lightning implementation </b>
1.To accommodate for variable modified in computation graph error, following hack has been applied - `self.target_model_loss_copy = self.target_model_loss.clone().detach() # is this hack correct`

<b> Tests to do </b>
* Check DPRNN val loss after one epoch and compare with vanilla implementation
    loss=-1


GANs here are not used to generate new data rather only as a min-max component to find the best augmentation policy.

Issues with Autoaugment - 

 * Learned policy is fixed for the entire training process. All possible instances of target example will be sent to the
    target network repeatedly, which will result in an inevitable overfitting in a long epoch training. 

Advantages of Adversarial Autoaugment  - 
    * Only one target network is used to evaluate the performance of augmentation policies. 
    * To combat harder examples from augmented by adversarial policies, target network has to learn more robust features. 


## Notes to understand code

* Hyperparameter used in the experiment. Represented as (Hyperparameter name, Value in code)
    - B(3) - Batch size
    - C(3) - Number of channels
    - H(32) - Height of the image
    - W(32) - Width of the image
    - M(8) - Number of policies by which each input example in the batch is augmented by
    - Q(5) - Number of sub policies in each of the M policies.

* `augment_dict` in `src/dataloader/augmentations.py` contains a dictionary of all augmentations for the experiment. 
* `parsed_policies[M, Q] ` in `src/dataloader/main.py` is a list of all policies.

```
parsed_policies[0] -> list of length Q
[
    [('Sharpness', 0.0), ('Sharpness', 0.7777777777777778)],
    [('Rotate', 0.2222222222222222), ('Sharpness', 0.0)],
    [('Color', 1.0), ('TranslateX', 1.0)],
    [('TranslateY', 0.5555555555555556), ('ShearY', 0.3333333333333333)],
    [('TranslateX', 0.2222222222222222), ('Contrast', 0.5555555555555556)]
]
```
Each sub-policy in `parsed_polices` consists of `Q` augmentation combinations as seen the example above. 
`parsed_policies` is input to `MultiAugmentation` class, which wraps each policy in `parsed_policies` with `Augmentation` class. `Augmentation` class selects one sub-policy (e.g. `[('TranslateY', 0.5555555555555556), ('ShearY', 0.3333333333333333)],`) from `parsed_policies[i]`.

Each item in the list `parsed_policies[i][j]` is a tuple`('Rotate', 0.2222222222222222)` with the name of the augmentation and the magnitude of the operation. To see how this magnitude is used please see the function `apply_augment` in `src/dataloader/augmentations.py`

Each image in the batch is augmented by `M` sub-policies, making the effective batch size `B * M`. Each sub-policy has two augmentation operations randomly selected from `parsed_policies[i]` which is a list of length `Q`. 

The maximum extent of the augmentation is already predefined in `augment_dict` in `src/dataloader/augmentations.py`.

* `trfs_list` is a list of augmentations
 `[RandomCrop(size=(32, 32), padding=4), RandomHorizontalFlip(p=0.5), Lambda(), Lambda()]` 

* `parsed_policies` are wrapped with `MultiAugmentation` class and appended to `trf_list`
    ```
    trfs_list[2].policies
    [
    <dataloader.transform.Augmentation object at 0x7fee73404df0>,
    <dataloader.transform.Augmentation object at 0x7fee73404e80>,
    <dataloader.transform.Augmentation object at 0x7fee73404f70>,
    <dataloader.transform.Augmentation object at 0x7fee73404fd0>,
    <dataloader.transform.Augmentation object at 0x7fee73441070>,
    <dataloader.transform.Augmentation object at 0x7fee734410d0>,
    <dataloader.transform.Augmentation object at 0x7fee73441130>,
    <dataloader.transform.Augmentation object at 0x7fee73441190>
    ]
    ```
    ```
    trfs_list[2].policies[0].policy
    [
    [('Sharpness', 0.0), ('Sharpness', 0.7777777777777778)],
    [('Rotate', 0.2222222222222222), ('Sharpness', 0.0)],
    [('Color', 1.0), ('TranslateX', 1.0)],
    [('TranslateY', 0.5555555555555556), ('ShearY', 0.3333333333333333)],
    [('TranslateX', 0.2222222222222222), ('Contrast', 0.5555555555555556)]
    ]
    ```

* Argument to ` with autocast(enabled=args.amp):` is `store_true` why is that so ? 

* `input.shape -> torch.Size([M * B, C, H, W])`. `M*B` to use `M` policies on the same batch. This is practically made
    possible by using `collate ` function which is defined in `pytorch` docs as -> `collate_fn (callable, optional): `merges a list of samples to form a mini-batch of Tensor(s).  Used when using batched loading from a map-style dataset. 

* Minimizing loss
 - `losses = [criterion(pred[i::args.M,...] ,label) for i in range(args.M)]` . `pred[i::args.M,...]` is of shape
     `[B, num_classes]`. `args.M` is the number of unique policies(`M`). 
 - In summary the model is given an input of `[M * B, C, H, W]`. And label of each of these `M` instances are the same; i.e label of `[0, B, C, H, W]` == `[[1...M], B, C, H, W]`

* In this code segment - `Lm[i] += reduced_metric(_loss.detach(), num_gpus, args.local_rank !=-1) / len(train_loader)`. Why divide by `len(train_loader)` which has a very high value of `8333`. 


* `trfs_list` in `main.py` is a pointer to `train_loader.dataset.dataset.transform.transforms` which initially has
    following list of objects - 
    ```
    [RandomCrop(size=(32, 32), padding=4), RandomHorizontalFlip(p=0.5), Lambda(), Lambda()]
    ```

