    # Mixed precision training parameters
    parser.add_argument('--apex', dest='apex', action='store_true',
                        help='Use apex for mixed precision training')
    parser.add_argument('--apex-opt-level', dest='apex_opt_level', default='O1', type=str,
                        help='For apex mixed precision training'
                             'O0 for FP32 training, O1 for mixed precision training.'
                             'For further detail, see https://github.com/NVIDIA/apex/tree/master/examples/imagenet')
    parser.add_argument('--sync-bn', dest='sync_bn', action='store_true',
                        help='Use sync batch norm')
    parser.add_argument('--cache-dataset', dest='cache_dataset', action='store_true',
                        help='Cache the datasets for quicker initialization. It also serializes the transforms')
    parser.add_argument('--print-freq', dest='print_freq', default=10, type=int, metavar='N',
                        help='print frequency')
    
    