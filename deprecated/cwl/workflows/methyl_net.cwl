class: Workflow
cwlVersion: v1.0
id: methyl_net
label: methyl_net
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
inputs:
  - id: query
    type: string
    'sbg:x': -108.6614761352539
    'sbg:y': -35.01996612548828
  - id: include_columns_file
    type: File?
    'sbg:x': 89.87889099121094
    'sbg:y': -74.16876983642578
  - id: disease_class_column
    type: string?
    'sbg:x': 95.18537139892578
    'sbg:y': 165.68411254882812
  - id: n_neighbors
    type: int?
    'sbg:x': 852.6881713867188
    'sbg:y': -156.36669921875
  - id: n_top_cpgs
    type: int?
    'sbg:x': 1103.4071044921875
    'sbg:y': -121.16587829589844
  - id: n_latent
    type: int?
    'sbg:x': 1326.7222900390625
    'sbg:y': -137.82904052734375
  - id: n_neighbor
    type: int?
    'sbg:x': 1610.08251953125
    'sbg:y': -127.23672485351562
  - id: cuda
    type: boolean?
    'sbg:exposed': true
  - id: learning_rate
    type: float?
    'sbg:exposed': true
  - id: weight_decay
    type: float?
    'sbg:exposed': true
  - id: n_epochs
    type: int?
    'sbg:exposed': true
  - id: hidden_layer_encoder_topology
    type: string?
    'sbg:exposed': true
  - id: column_of_interest_1
    type: string?
    'sbg:exposed': true
  - id: axes_off_1
    type: boolean?
    'sbg:exposed': true
  - id: supervised_1
    type: boolean?
    'sbg:exposed': true
  - id: n_cores
    type: int?
    'sbg:exposed': true
  - id: meffil
    type: boolean?
    'sbg:exposed': true
  - id: disease_only
    type: boolean?
    'sbg:exposed': true
  - id: subtype_delimiter
    type: string?
    'sbg:exposed': true
outputs:
  - id: output_visual
    outputSource:
      - transform_plot_1/output_visual
    type: File
    'sbg:x': 1972.8907470703125
    'sbg:y': -10.714360237121582
  - id: pytorch_model
    outputSource:
      - vae_embed/pytorch_model
    type: File?
    'sbg:x': 1477.431396484375
    'sbg:y': 21.976003646850586
steps:
  - id: create_sample_sheet
    in:
      - id: input_sample_sheet
        source: download_geo_1/initial_sample_sheet
      - id: source_type
        default: geo
      - id: idat_dir
        source: download_geo_1/idat_dir
      - id: header_line
        default: 0
      - id: disease_class_column
        source: disease_class_column
      - id: include_columns_file
        source: include_columns_file
    out:
      - id: idat_dir_out
      - id: final_csv
    run: ../tools/create_sample_sheet.cwl
    label: create_sample_sheet
    'sbg:x': 274.3111877441406
    'sbg:y': -175.69915771484375
  - id: meffil_encode_1
    in:
      - id: input_sample_sheet
        source: create_sample_sheet/final_csv
      - id: idat_dir_input
        source: create_sample_sheet/idat_dir_out
    out:
      - id: output_sample_sheet
      - id: idat_dir
    run: ../tools/meffil_encode.cwl
    label: meffil_encode
    'sbg:x': 413.171630859375
    'sbg:y': -98.21781921386719
  - id: imputation
    in:
      - id: input_pkl
        source: combine_methyl_arrays/output_methyl_array
      - id: imputation_method
        default: KNN
      - id: solver
        default: fancyimpute
      - id: n_neighbors
        source: n_neighbors
    out:
      - id: imputed_methylarray
    run: ../tools/imputation.cwl
    label: imputation
    'sbg:x': 965.3180541992188
    'sbg:y': 86.1338119506836
  - id: mad_filter
    in:
      - id: input_pkl
        source: imputation/imputed_methylarray
      - id: n_top_cpgs
        source: n_top_cpgs
    out:
      - id: output_pkl
    run: ../tools/mad_filter.cwl
    label: mad_filter
    'sbg:x': 1090.9664306640625
    'sbg:y': 105.3855972290039
  - id: vae_embed
    in:
      - id: input_pkl
        source: mad_filter/output_pkl
      - id: cuda
        source: cuda
      - id: n_latent
        source: n_latent
      - id: learning_rate
        source: learning_rate
      - id: weight_decay
        source: weight_decay
      - id: n_epochs
        source: n_epochs
      - id: hidden_layer_encoder_topology
        source: hidden_layer_encoder_topology
    out:
      - id: output_methyl_array_encoded
      - id: pytorch_model
    run: ../tools/vae_embed.cwl
    label: vae_embed
    'sbg:x': 1353.5640869140625
    'sbg:y': 113.09715270996094
  - id: transform_plot_1
    in:
      - id: input_pkl
        source:
          - vae_embed/output_methyl_array_encoded
          - mad_filter/output_pkl
      - id: column_of_interest
        default: disease
        source: column_of_interest_1
      - id: n_neighbor
        source: n_neighbor
      - id: axes_off
        source: axes_off_1
      - id: supervised
        source: supervised_1
    out:
      - id: output_visual
    run: ../tools/transform_plot.cwl
    label: transform_plot
    scatter:
      - input_pkl
    scatterMethod: dotproduct
    'sbg:x': 1703.8648681640625
    'sbg:y': 18.198997497558594
  - id: download_geo_1
    in:
      - id: query
        source: query
    out:
      - id: idat_dir
      - id: initial_sample_sheet
    run: ../tools/download_geo.cwl
    label: download_geo
    'sbg:x': 55.04618453979492
    'sbg:y': -248.47854614257812
  - id: preprocess_pipeline
    in:
      - id: idat_csv_dir_input
        source: split_by_subtype/output_dirs
      - id: idat_dir
        source: meffil_encode_1/idat_dir
      - id: n_cores
        default: 6
        source: n_cores
      - id: meffil
        default: true
        source: meffil
    out:
      - id: output_pkl
    run: ../tools/preprocess_pipeline.cwl
    label: preprocess_pipeline
    scatter:
      - idat_csv_dir_input
    scatterMethod: dotproduct
    'sbg:x': 695.3563842773438
    'sbg:y': 90.52800750732422
  - id: split_by_subtype
    in:
      - id: idat_csv
        source: meffil_encode_1/output_sample_sheet
      - id: disease_only
        default: false
        source: disease_only
      - id: subtype_delimiter
        default: ','
        source: subtype_delimiter
    out:
      - id: output_dirs
    run: ../tools/split_by_subtype.cwl
    label: split_by_subtype
    'sbg:x': 515.5643920898438
    'sbg:y': 56.349822998046875
  - id: combine_methyl_arrays
    in:
      - id: input_pkls
        linkMerge: merge_flattened
        source:
          - preprocess_pipeline/output_pkl
    out:
      - id: output_methyl_array
    run: ../tools/combine_methyl_arrays.cwl
    label: combine_methyl_arrays
    'sbg:x': 837.650146484375
    'sbg:y': 113.59088134765625
requirements:
  - class: ScatterFeatureRequirement
  - class: MultipleInputFeatureRequirement
