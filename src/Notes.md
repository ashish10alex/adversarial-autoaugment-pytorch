
## Notes to understand code

Hyperparameter used in the experiment. Represented as (Hyperparameter name, Value in code)
* B(6) - Batch size
* C(3) - Number of channels
* H(32) - Height of the image
* W(32) - Width of the image
* M(8) - Number of instances of input examples augmented by adversarial examples. 
* Q(5) - Number of sub policies in each of the M policies.

* `parsed_policies [M, Q]` is a list of augmentations 

```
parsed_policies[0]
[
    [('Sharpness', 0.0), ('Sharpness', 0.7777777777777778)],
    [('Rotate', 0.2222222222222222), ('Sharpness', 0.0)],
    [('Color', 1.0), ('TranslateX', 1.0)],
    [('TranslateY', 0.5555555555555556), ('ShearY', 0.3333333333333333)],
    [('TranslateX', 0.2222222222222222), ('Contrast', 0.5555555555555556)]
]
```

* `trfs_list` is a list of augmentations
 `[RandomCrop(size=(32, 32), padding=4), RandomHorizontalFlip(p=0.5), Lambda(), Lambda()]` 

* `parsed_polcies` are wrapped with `MultiAugmentation` class and appended to `trf_list`
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
    possible by using `collate ` function which is defined in `pytorch` docs as -> `collate_fn (callable, optional): `merges a
list of samples to form a mini-batch of Tensor(s).  Used when using batched loading from a map-style dataset. 

* Minimizing loss
 - `losses = [criterion(pred[i::args.M,...] ,label) for i in range(args.M)]` . `pred[i::args.M,...]` is of shape
     `[B, num_classes]`. `args.M` is the number of unique policies(`M`). 
 - In summary the model is given an input of `[M * B, C, H, W]`. And label of each of these `M` instances are the same; i.e label of `[0, B, C, H, W]` == `[[1...M], B, C, H, W]`

* In this code segment - `Lm[i] += reduced_metric(_loss.detach(), num_gpus, args.local_rank !=-1) / len(train_loader)`.
    Why divide by `len(train_loader)` which has a very high value of `8333`. 
