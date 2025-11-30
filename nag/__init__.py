from tools.util.reflection import _register_legacy_import
legacy_names = {
    "renamings" : [
        {
            "old_import" : "nsf.run.scene_nsf_runner.SceneNSFRunner",
            "new_import" : "nag.run.nag_runner.NAGRunner"
        },
        {
            "old_import" : "nsf.dataset.scene_nsf_bundle_dataset.SceneNSFBundleDataset",
            "new_import" : "nag.dataset.nag_dataset.NAGDataset"
        },
        {
            "old_import" : "nsf.dataset.scene_nsf_bundle_dataset",
            "new_import" : "nag.dataset.nag_dataset"
        },
        {
            "old_import" : "nsf.model.scene_nsf_model.SceneNSFModel",
            "new_import" : "nag.model.nag_model.NAGModel"
        },
        {
            "old_import" : "nsf.model.scene_nsf_model",
            "new_import" : "nag.model.nag_model"
        },
        {
            "old_import" : "nsf.model.scene_nsf_functional_model.SceneNSFFunctionalModel",
            "new_import" : "nag.model.nag_functional_model.NAGFunctionalModel"
        },
        {
            "old_import" : "nsf.model.scene_nsf_functional_model",
            "new_import" : "nag.model.nag_functional_model"
        },
        {
            "old_import" : "nsf.callbacks.scene_nsf_callback.SceneNSFCallback",
            "new_import" : "nag.callbacks.nag_callback.NAGCallback"
        },
        {
            "old_import" : "nsf.callbacks.scene_nsf_callback",
            "new_import" : "nag.callbacks.nag_callback"
        },
        {
            "old_import" : "nsf",
            "new_import" : "nag"
        },
    ],
}
for renaming in legacy_names["renamings"]:
    _register_legacy_import(renaming["old_import"], renaming["new_import"])