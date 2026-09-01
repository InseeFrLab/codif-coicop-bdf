from typing import Dict, List, Optional, Tuple

def truncate_code(code: str, level: int) -> Optional[str]:
    """
    Truncate code to specified hierarchical level
    
    Args:
        code: Full code (e.g., '08.1.2.3.4' or '08.1.6')
        level: Level to truncate to (1-5)
    
    Returns:
        Truncated code or original if already at or below target level,
        None if invalid
    """
    if code is None or not isinstance(code, str) or code == '':
        return None
    
    # Split by dot separator
    parts = code.split('.')
    
    # If code is already at or below target level, return as-is
    if len(parts) <= level:
        return code
    
    # Otherwise truncate to target level
    return '.'.join(parts[:level])


def check_label_in_retrieved(
    label_code: str,
    retrieved_codes: List[str],
    level: int
) -> bool:
    """
    Check if the label code is present in the retrieved codes list at a given level
    
    Args:
        label_code: Ground truth code
        retrieved_codes: List of retrieved codes from RAG
        level: Hierarchical level (1-5) to check
    
    Returns:
        True if label is in retrieved codes at this level, False otherwise
    """
    if label_code is None or retrieved_codes is None:
        return False
    
    # Truncate label to specified level
    label_truncated = truncate_code(label_code, level)
    if label_truncated is None:
        return False
    
    # Check if any retrieved code matches at this level
    for retrieved_code in retrieved_codes:
        retrieved_truncated = truncate_code(retrieved_code, level)
        if retrieved_truncated == label_truncated:
            return True
    
    return False



def calculate_accuracy_at_level(
    records: List[Dict],
    predicted_col: str,
    label_col: str,
    level: int,
    retrieved_col: str = 'list_retrieved_codes'
) -> Tuple[float, List[bool], float, float, List[bool]]:
    """
    Calculate accuracy at a specific hierarchical level with retrieval analysis
    
    Args:
        records: List of dictionaries with predictions and labels
        predicted_col: Key name for predicted code
        label_col: Key name for labeled code
        level: Hierarchical level (1-5)
        retrieved_col: Key name for list of retrieved codes
    
    Returns:
        Tuple containing:
        - overall_accuracy: Overall accuracy (0.0 to 1.0)
        - result_list: List of bool indicating if each prediction is correct
        - retrieval_accuracy: Proportion of cases where label is in retrieved codes
        - generation_accuracy_when_retrieved: Accuracy when label is in retrieved codes
        - label_in_retrieved_list: List of bool indicating if label is in retrieved codes
    """
    correct = 0
    total = 0
    result_list = []
    label_in_retrieved_list = []
    
    # For generation accuracy when retrieved
    correct_when_retrieved = 0
    total_when_retrieved = 0
    
    for record in records:
        pred_code = record.get(predicted_col)
        label_code = record.get(label_col)
        retrieved_codes = record.get(retrieved_col, [])
        
        # Truncate codes to specified level
        pred_truncated = truncate_code(pred_code, level)
        label_truncated = truncate_code(label_code, level)
        
        # Skip if either truncation failed
        # if pred_truncated is None or label_truncated is None:
        #     result_list.append(False)
        #     label_in_retrieved_list.append(False)
        #     continue
        
        # Check if prediction is correct
        is_correct = (pred_truncated == label_truncated)
        result_list.append(is_correct)
        
        # Check if label is in retrieved codes
        label_is_retrieved = check_label_in_retrieved(
            label_code, 
            retrieved_codes, 
            level
        )
        label_in_retrieved_list.append(label_is_retrieved)
        
        # Update overall accuracy counters
        total += 1
        if is_correct:
            correct += 1
        
        # Update generation accuracy when retrieved counters
        if label_is_retrieved:
            total_when_retrieved += 1
            if is_correct:
                correct_when_retrieved += 1
    
    # Calculate accuracies
    overall_accuracy = correct / total if total > 0 else 0.0
    retrieval_accuracy = sum(label_in_retrieved_list) / len(label_in_retrieved_list) if len(label_in_retrieved_list) > 0 else 0.0
    generation_accuracy_when_retrieved = correct_when_retrieved / total_when_retrieved if total_when_retrieved > 0 else 0.0
    
    return (
        overall_accuracy,
        result_list,
        retrieval_accuracy,
        generation_accuracy_when_retrieved,
        label_in_retrieved_list
    )


def filter_records(
    records: List[Dict],
    parsed_col: str,
    codable_col: str,
    filter_type: str,
    confidence_col: str = "confidence",
    threshold: float = 0.7,
) -> List[Dict]:
    """
    Filter records based on filter type
    
    Args:
        records: List of all records
        parsed_col: Key name for parsing flag
        codable_col: Key name for codability flag
        filter_type: One of 'all_raw', 'all_parsed', 'codable_only', 'parsed_and_codable'
    
    Returns:
        Filtered list of records
    """
    if filter_type == 'all_raw':
        return records
    
    elif filter_type == 'all_parsed':
        return [r for r in records if r.get(parsed_col) == True]
    
    elif filter_type == 'codable_only':
        return [r for r in records if r.get(codable_col) == True]
    
    elif filter_type == 'parsed_and_codable':
        return [
            r for r in records 
            if r.get(parsed_col) == True and r.get(codable_col) == True
        ]
    elif filter_type == 'threshold':
        return [
            r for r in records 
            if r.get(parsed_col) == True and r.get(codable_col) == True and r.get(confidence_col) >= threshold 
        ]
    
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")





# def compute_hierarchical_metrics(
#     records: List[Dict],
#     product_col: str = "product",
#     predicted_col: str = "code_predict",
#     label_col: str = "code",
#     confidence_col: str = "confidence",
#     codable_col: str = "codable",
#     parsed_col: str = "parsed",
#     retrieved_col: str = "list_retrieved_codes",
#     threshold: float = 0.7,
# ) -> Dict[str, Dict]:
#     """
#     Compute hierarchical accuracy metrics with retrieval analysis
    
#     Args:
#         records: List of dictionnaries
#         product_col: Column name for product description
#         predicted_col: Column name for predicted code
#         label_col: Column name for labeled/ground truth code
#         confidence_col: Column name for LLM confidence score
#         codable_col: Column name for codability flag (True/False)
#         parsed_col: Column name for parsing success flag (True/False)
#         retrieved_col: Column name for list of retrieved codes
    
#     Returns:
#         Dictionary with metrics including retrieval analysis:
#         {
#             'all_raw': {
#                 'level_1': float,
#                 'level_1_retrieval_accuracy': float,
#                 'level_1_generation_accuracy_when_retrieved': float,
#                 ...
#             },
#             'all_parsed': {...},
#             'codable_only': {...},
#             'parsed_and_codable': {...}
#         }
#     """
    
#     # Initialize results dictionary
#     results = {
#         'all_raw': {},
#         'all_parsed': {},
#         'codable_only': {},
#         'parsed_and_codable': {},
#         'threshold': {},
#     }
    
#     # Define filter types
#     filter_types = ['all_raw', 'all_parsed', 'codable_only', 'parsed_and_codable', 'threshold']
    
#     # Calculate metrics for each filter type
#     for filter_type in filter_types:
#         # Filter records according to type
#         filtered_records = filter_records(
#             records, 
#             parsed_col, 
#             codable_col, 
#             filter_type,
#             confidence_col,
#             threshold
#         )
        
#         # Store number of samples
#         results[filter_type]['n_samples'] = len(filtered_records)
        
#         # Calculate accuracy at each hierarchical level
#         for level in range(1, 6):
#             (
#                 overall_acc,
#                 result_list,
#                 retrieval_acc,
#                 generation_acc_when_retrieved,
#                 label_in_retrieved_list
#             ) = calculate_accuracy_at_level(
#                 filtered_records,
#                 predicted_col,
#                 label_col,
#                 level,
#                 retrieved_col
#             )
            
#             # Store all metrics
#             results[filter_type][f'level_{level}'] = overall_acc
#             results[filter_type][f'level_{level}_retrieval_accuracy'] = retrieval_acc
#             results[filter_type][f'level_{level}_generation_accuracy_when_retrieved'] = generation_acc_when_retrieved
    
#     return results


def compute_hierarchical_metrics(
    records: List[Dict],
    product_col: str = "product",
    predicted_col: str = "code_predict",
    label_col: str = "code",
    confidence_col: str = "confidence",
    codable_col: str = "codable",
    parsed_col: str = "parsed",
    retrieved_col: str = "list_retrieved_codes",
    threshold: float = 0.7,
    by_product_type: bool = True,
) -> Dict[str, Dict]:
    """
    Compute hierarchical accuracy metrics with retrieval analysis.

    Optimized: truncations and retrieval checks are pre-computed once per record,
    and filter groups are built once and reused across product types.

    Args:
        records: List of dictionnaries
        product_col: Column name for product description
        predicted_col: Column name for predicted code
        label_col: Column name for labeled/ground truth code
        confidence_col: Column name for LLM confidence score
        codable_col: Column name for codability flag (True/False)
        parsed_col: Column name for parsing success flag (True/False)
        retrieved_col: Column name for list of retrieved codes
        threshold: Confidence threshold for filtering
        by_product_type: If True, compute metrics per product type (COICOP prefix)

    Returns:
        Dictionary with metrics including retrieval analysis and optional breakdown by product type
    """

    # ------------------------------------------------------------------
    # 1. Pre-process: truncate codes and check retrieval at all levels once
    # ------------------------------------------------------------------
    preprocessed = []
    for r in records:
        pred = r.get(predicted_col)
        label = r.get(label_col)
        retrieved = r.get(retrieved_col) or []

        pred_trunc  = [truncate_code(pred,  lvl) for lvl in range(1, 6)]
        label_trunc = [truncate_code(label, lvl) for lvl in range(1, 6)]
        label_in_ret = [check_label_in_retrieved(label, retrieved, lvl) for lvl in range(1, 6)]

        preprocessed.append({
            "pred_trunc":    pred_trunc,
            "label_trunc":   label_trunc,
            "label_in_ret":  label_in_ret,
            "is_parsed":     r.get(parsed_col) == True,
            "is_codable":    r.get(codable_col) == True,
            "confidence":    r.get(confidence_col) or 0,
            "label_prefix":  str(label)[:2] if label else None,
        })

    # ------------------------------------------------------------------
    # 2. Pre-build filter groups once
    # ------------------------------------------------------------------
    filter_groups = {
        "all_raw":            preprocessed,
        "all_parsed":         [p for p in preprocessed if p["is_parsed"]],
        "codable_only":       [p for p in preprocessed if p["is_codable"]],
        "parsed_and_codable": [p for p in preprocessed if p["is_parsed"] and p["is_codable"]],
        "threshold":          [p for p in preprocessed if p["is_parsed"] and p["is_codable"] and p["confidence"] >= threshold],
    }

    # ------------------------------------------------------------------
    # 3. Helper: single pass over a group, all levels at once
    # ------------------------------------------------------------------
    def _metrics_for_group(group: list) -> dict:
        result = {"n_samples": len(group)}
        for l_idx in range(5):
            correct = total = correct_when_ret = total_when_ret = 0
            for p in group:
                is_correct = p["pred_trunc"][l_idx] == p["label_trunc"][l_idx]
                in_ret = p["label_in_ret"][l_idx]
                total += 1
                correct += is_correct
                if in_ret:
                    total_when_ret += 1
                    correct_when_ret += is_correct
            lvl = l_idx + 1
            result[f"level_{lvl}"] = correct / total if total else 0.0
            result[f"level_{lvl}_retrieval_accuracy"] = (
                sum(p["label_in_ret"][l_idx] for p in group) / len(group) if group else 0.0
            )
            result[f"level_{lvl}_generation_accuracy_when_retrieved"] = (
                correct_when_ret / total_when_ret if total_when_ret else 0.0
            )
        return result

    # ------------------------------------------------------------------
    # 4. Overall metrics
    # ------------------------------------------------------------------
    results: Dict = {"overall": {}, "by_product_type": {}}
    for filter_type, group in filter_groups.items():
        results["overall"][filter_type] = _metrics_for_group(group)

    # ------------------------------------------------------------------
    # 5. By product type (reuse pre-filtered groups)
    # ------------------------------------------------------------------
    if by_product_type:
        product_types = sorted(
            {p["label_prefix"] for p in preprocessed if p["label_prefix"]}
        )
        for product_type in product_types:
            results["by_product_type"][product_type] = {}
            for filter_type, group in filter_groups.items():
                pt_group = [p for p in group if p["label_prefix"] == product_type]
                results["by_product_type"][product_type][filter_type] = (
                    _metrics_for_group(pt_group) if pt_group
                    else {
                        "n_samples": 0,
                        **{
                            k: None
                            for lvl in range(1, 6)
                            for k in (
                                f"level_{lvl}",
                                f"level_{lvl}_retrieval_accuracy",
                                f"level_{lvl}_generation_accuracy_when_retrieved",
                            )
                        },
                    }
                )

    return results




def print_metrics_report(metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted report of the metrics including retrieval analysis
    
    Args:
        metrics: Dictionary returned by compute_hierarchical_metrics
    """
    print("=" * 100)
    print("HIERARCHICAL CLASSIFICATION METRICS WITH RETRIEVAL ANALYSIS")
    print("=" * 100)
    
    for metric_type, values in metrics.items():
        print(f"\n{'─' * 100}")
        print(f"Metric Type: {metric_type.upper().replace('_', ' ')}")
        print(f"{'─' * 100}")
        print(f"Number of samples: {values['n_samples']}")
        print()
        print(f"{'Level':<8} {'Overall Acc':<15} {'Retrieval Acc':<18} {'Gen Acc (Retrieved)':<20}")
        print(f"{'-'*8} {'-'*15} {'-'*18} {'-'*20}")
        
        for level in range(1, 6):
            overall_acc = values[f'level_{level}']
            retrieval_acc = values[f'level_{level}_retrieval_accuracy']
            gen_acc = values[f'level_{level}_generation_accuracy_when_retrieved']
            
            print(
                f"{level:<8} "
                f"{overall_acc:<15.4f} "
                f"{retrieval_acc:<18.4f} "
                f"{gen_acc:<20.4f}"
            )
    
    print("\n" + "=" * 100)

# def write_metrics_report(
#     metrics: Dict[str, Dict[str, float]], 
#     output_path: str
# ) -> None:
#     """
#     Write a formatted report of the metrics including retrieval analysis to a text file
    
#     Args:
#         metrics: Dictionary returned by compute_hierarchical_metrics
#         output_path: Path to the output .txt file
#     """
#     with open(output_path, 'w') as f:
#         f.write("=" * 100 + "\n")
#         f.write("HIERARCHICAL CLASSIFICATION METRICS WITH RETRIEVAL ANALYSIS\n")
#         f.write("=" * 100 + "\n")
        
#         for metric_type, values in metrics.items():
#             f.write(f"\n{'─' * 100}\n")
#             f.write(f"Metric Type: {metric_type.upper().replace('_', ' ')}\n")
#             f.write(f"{'─' * 100}\n")
#             f.write(f"Number of samples: {values['n_samples']}\n")
#             f.write("\n")
#             f.write(f"{'Level':<8} {'Overall Acc':<15} {'Retrieval Acc':<18} {'Gen Acc (Retrieved)':<20}\n")
#             f.write(f"{'-'*8} {'-'*15} {'-'*18} {'-'*20}\n")
            
#             for level in range(1, 6):
#                 overall_acc = values[f'level_{level}']
#                 retrieval_acc = values[f'level_{level}_retrieval_accuracy']
#                 gen_acc = values[f'level_{level}_generation_accuracy_when_retrieved']
                
#                 f.write(
#                     f"{level:<8} "
#                     f"{overall_acc:<15.4f} "
#                     f"{retrieval_acc:<18.4f} "
#                     f"{gen_acc:<20.4f}\n"
#                 )
        
#         f.write("\n" + "=" * 100 + "\n")


def write_metrics_report(
    metrics: Dict[str, Dict[str, float]],
    output_path: str,
    include_product_types: bool = True,
    include_comparison: bool = True,
    by_nature_metrics: Optional[Dict[str, Dict]] = None,
) -> None:
    """
    Write a formatted report of the metrics including retrieval analysis to a text file
    
    Args:
        metrics: Dictionary returned by compute_hierarchical_metrics
        output_path: Path to the output .txt file
        include_product_types: If True, include detailed metrics for each product type
        include_comparison: If True, include comparison tables across product types
        by_nature_metrics: Optional dict {nature: metrics_dict} for per-nature breakdown.
            Only written when multiple natures are present (caller's responsibility).
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("HIERARCHICAL CLASSIFICATION METRICS WITH RETRIEVAL ANALYSIS\n")
        f.write("=" * 100 + "\n")
        
        # ========== OVERALL METRICS ==========
        f.write("\n")
        f.write("█" * 100 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("█" * 100 + "\n")
        
        for metric_type, values in metrics['overall'].items():
            f.write(f"\n{'─' * 100}\n")
            f.write(f"Metric Type: {metric_type.upper().replace('_', ' ')}\n")
            f.write(f"{'─' * 100}\n")
            f.write(f"Number of samples: {values['n_samples']}\n")
            f.write("\n")
            f.write(f"{'Level':<8} {'Overall Acc':<15} {'Retrieval Acc':<18} {'Gen Acc (Retrieved)':<20}\n")
            f.write(f"{'-'*8} {'-'*15} {'-'*18} {'-'*20}\n")
            
            for level in range(1, 6):
                overall_acc = values[f'level_{level}']
                retrieval_acc = values[f'level_{level}_retrieval_accuracy']
                gen_acc = values[f'level_{level}_generation_accuracy_when_retrieved']
                
                f.write(
                    f"{level:<8} "
                    f"{overall_acc:<15.4f} "
                    f"{retrieval_acc:<18.4f} "
                    f"{gen_acc:<20.4f}\n"
                )
        
        # ========== BY ANNOTATION NATURE ==========
        if by_nature_metrics:
            f.write("\n\n")
            f.write("█" * 100 + "\n")
            f.write("OVERALL METRICS BY ANNOTATION NATURE\n")
            f.write("█" * 100 + "\n")

            for nature, nature_metrics in sorted(by_nature_metrics.items()):
                f.write(f"\n{'▓' * 100}\n")
                f.write(f"NATURE: {nature}\n")
                f.write(f"{'▓' * 100}\n")

                for filter_type, values in nature_metrics["overall"].items():
                    f.write(f"\n{'─' * 100}\n")
                    f.write(f"Metric Type: {filter_type.upper().replace('_', ' ')}\n")
                    f.write(f"{'─' * 100}\n")
                    f.write(f"Number of samples: {values['n_samples']}\n")
                    f.write("\n")
                    f.write(f"{'Level':<8} {'Overall Acc':<15} {'Retrieval Acc':<18} {'Gen Acc (Retrieved)':<20}\n")
                    f.write(f"{'-'*8} {'-'*15} {'-'*18} {'-'*20}\n")

                    for level in range(1, 6):
                        overall_acc = values[f'level_{level}']
                        retrieval_acc = values[f'level_{level}_retrieval_accuracy']
                        gen_acc = values[f'level_{level}_generation_accuracy_when_retrieved']
                        f.write(
                            f"{level:<8} "
                            f"{overall_acc:<15.4f} "
                            f"{retrieval_acc:<18.4f} "
                            f"{gen_acc:<20.4f}\n"
                        )

        # ========== PRODUCT TYPE COMPARISON ==========
        if include_comparison and 'by_product_type' in metrics and metrics['by_product_type']:
            f.write("\n\n")
            f.write("█" * 100 + "\n")
            f.write("PRODUCT TYPE COMPARISON\n")
            f.write("█" * 100 + "\n")
            
            # For each filter type
            for filter_type in ['all_raw', 'all_parsed', 'codable_only', 'parsed_and_codable', 'threshold']:
                f.write(f"\n{'═' * 100}\n")
                f.write(f"Filter: {filter_type.upper().replace('_', ' ')}\n")
                f.write(f"{'═' * 100}\n")
                
                # For each level
                for level in range(1, 6):
                    f.write(f"\n--- Level {level} ---\n")
                    f.write(f"{'Type':<8} {'N':<8} {'Overall':<12} {'Retrieval':<12} {'Gen|Retr':<12}\n")
                    f.write(f"{'-'*8} {'-'*8} {'-'*12} {'-'*12} {'-'*12}\n")
                    
                    # Collect data for sorting
                    type_data = []
                    for product_type in sorted(metrics['by_product_type'].keys()):
                        if filter_type in metrics['by_product_type'][product_type]:
                            type_metrics = metrics['by_product_type'][product_type][filter_type]
                            n_samples = type_metrics['n_samples']
                            acc = type_metrics.get(f'level_{level}')
                            ret_acc = type_metrics.get(f'level_{level}_retrieval_accuracy')
                            gen_acc = type_metrics.get(f'level_{level}_generation_accuracy_when_retrieved')
                            
                            if acc is not None and n_samples > 0:
                                type_data.append({
                                    'type': product_type,
                                    'n': n_samples,
                                    'acc': acc,
                                    'ret': ret_acc,
                                    'gen': gen_acc
                                })
                    
                    # Sort by accuracy
                    type_data.sort(key=lambda x: x['acc'], reverse=True)
                    
                    # Write sorted data
                    for data in type_data:
                        f.write(
                            f"{data['type']:<8} "
                            f"{data['n']:<8} "
                            f"{data['acc']:<12.4f} "
                            f"{data['ret']:<12.4f} "
                            f"{data['gen']:<12.4f}\n"
                        )
                    
                    if not type_data:
                        f.write("No data available\n")
        
        # ========== DETAILED METRICS BY PRODUCT TYPE ==========
        if include_product_types and 'by_product_type' in metrics and metrics['by_product_type']:
            f.write("\n\n")
            f.write("█" * 100 + "\n")
            f.write("DETAILED METRICS BY PRODUCT TYPE\n")
            f.write("█" * 100 + "\n")
            
            for product_type in sorted(metrics['by_product_type'].keys()):
                f.write(f"\n\n{'▓' * 100}\n")
                f.write(f"PRODUCT TYPE: {product_type}\n")
                f.write(f"{'▓' * 100}\n")
                
                for metric_type, values in metrics['by_product_type'][product_type].items():
                    f.write(f"\n{'─' * 100}\n")
                    f.write(f"Metric Type: {metric_type.upper().replace('_', ' ')}\n")
                    f.write(f"{'─' * 100}\n")
                    f.write(f"Number of samples: {values['n_samples']}\n")
                    
                    if values['n_samples'] > 0:
                        f.write("\n")
                        f.write(f"{'Level':<8} {'Overall Acc':<15} {'Retrieval Acc':<18} {'Gen Acc (Retrieved)':<20}\n")
                        f.write(f"{'-'*8} {'-'*15} {'-'*18} {'-'*20}\n")
                        
                        for level in range(1, 6):
                            overall_acc = values.get(f'level_{level}')
                            retrieval_acc = values.get(f'level_{level}_retrieval_accuracy')
                            gen_acc = values.get(f'level_{level}_generation_accuracy_when_retrieved')
                            
                            if overall_acc is not None:
                                f.write(
                                    f"{level:<8} "
                                    f"{overall_acc:<15.4f} "
                                    f"{retrieval_acc:<18.4f} "
                                    f"{gen_acc:<20.4f}\n"
                                )
                            else:
                                f.write(f"{level:<8} {'N/A':<15} {'N/A':<18} {'N/A':<20}\n")
                    else:
                        f.write("\nNo samples for this metric type.\n")
        
        # ========== SUMMARY STATISTICS ==========
        if 'by_product_type' in metrics and metrics['by_product_type']:
            f.write("\n\n")
            f.write("█" * 100 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("█" * 100 + "\n")
            
            # Calculate overall statistics
            total_types = len(metrics['by_product_type'])
            f.write(f"\nTotal product types: {total_types}\n")
            
            # Sample distribution
            f.write("\n--- Sample Distribution ---\n")
            f.write(f"{'Type':<8} {'All Raw':<12} {'All Parsed':<12} {'Codable':<12} {'P&C':<12} {'Threshold':<12}\n")
            f.write(f"{'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}\n")
            
            for product_type in sorted(metrics['by_product_type'].keys()):
                type_metrics = metrics['by_product_type'][product_type]
                f.write(f"{product_type:<8} ")
                for filter_type in ['all_raw', 'all_parsed', 'codable_only', 'parsed_and_codable', 'threshold']:
                    if filter_type in type_metrics:
                        n = type_metrics[filter_type]['n_samples']
                        f.write(f"{n:<12} ")
                    else:
                        f.write(f"{'N/A':<12} ")
                f.write("\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")

def export_metrics_to_list(metrics: Dict[str, Dict[str, float]]) -> List[Dict]:
    """
    Export metrics to a list of dictionaries for easy analysis
    
    Args:
        metrics: Dictionary returned by compute_hierarchical_metrics
    
    Returns:
        List of dictionaries with metrics in tabular format
    """
    rows = []
    
    for metric_type, values in metrics.items():
        for level in range(1, 6):
            rows.append({
                'metric_type': metric_type,
                'level': level,
                'overall_accuracy': values[f'level_{level}'],
                'retrieval_accuracy': values[f'level_{level}_retrieval_accuracy'],
                'generation_accuracy_when_retrieved': values[f'level_{level}_generation_accuracy_when_retrieved'],
                'n_samples': values['n_samples']
            })
    
    return rows


def analyze_error_sources(metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Analyze the proportion of errors due to retrieval vs generation
    
    Args:
        metrics: Dictionary returned by compute_hierarchical_metrics
    
    Returns:
        Dictionary with error analysis for each metric type and level
    """
    error_analysis = {}
    
    for metric_type, values in metrics.items():
        error_analysis[metric_type] = {}
        
        for level in range(1, 6):
            overall_acc = values[f'level_{level}']
            retrieval_acc = values[f'level_{level}_retrieval_accuracy']
            gen_acc_when_retrieved = values[f'level_{level}_generation_accuracy_when_retrieved']
            
            # Calculate error rates
            overall_error_rate = 1 - overall_acc
            
            # Retrieval errors: cases where label is NOT in retrieved codes
            retrieval_error_rate = 1 - retrieval_acc
            
            # Generation errors when retrieved: label IS in retrieved but prediction wrong
            # This is: (retrieval_acc * (1 - gen_acc_when_retrieved))
            generation_error_rate_when_retrieved = retrieval_acc * (1 - gen_acc_when_retrieved)
            
            error_analysis[metric_type][f'level_{level}'] = {
                'overall_error_rate': overall_error_rate,
                'retrieval_error_rate': retrieval_error_rate,
                'generation_error_rate_when_retrieved': generation_error_rate_when_retrieved,
                'retrieval_error_proportion': retrieval_error_rate / overall_error_rate if overall_error_rate > 0 else 0.0,
                'generation_error_proportion': generation_error_rate_when_retrieved / overall_error_rate if overall_error_rate > 0 else 0.0
            }
    
    return error_analysis


def print_error_analysis(error_analysis: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """
    Print error analysis showing proportion of retrieval vs generation errors
    
    Args:
        error_analysis: Dictionary returned by analyze_error_sources
    """
    print("\n" + "=" * 100)
    print("ERROR SOURCE ANALYSIS")
    print("=" * 100)
    
    for metric_type, values in error_analysis.items():
        print(f"\n{'─' * 100}")
        print(f"Metric Type: {metric_type.upper().replace('_', ' ')}")
        print(f"{'─' * 100}")
        print(f"{'Level':<8} {'Overall Err':<15} {'Retrieval Err':<18} {'Generation Err':<18} {'% Retrieval':<15} {'% Generation':<15}")
        print(f"{'-'*8} {'-'*15} {'-'*18} {'-'*18} {'-'*15} {'-'*15}")
        
        for level in range(1, 6):
            level_data = values[f'level_{level}']
            
            print(
                f"{level:<8} "
                f"{level_data['overall_error_rate']:<15.4f} "
                f"{level_data['retrieval_error_rate']:<18.4f} "
                f"{level_data['generation_error_rate_when_retrieved']:<18.4f} "
                f"{level_data['retrieval_error_proportion']:<15.2%} "
                f"{level_data['generation_error_proportion']:<15.2%}"
            )
    
    print("\n" + "=" * 100)


import mlflow
import tempfile
import contextlib
from typing import Dict

def save_metrics_report_as_artifact(metrics: Dict[str, Dict[str, float]], output_path: str = "report.txt"):
    """
    Call print_metrics_report, capture its output, and save it as an MLflow artifact.
    """
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp_file:
        filename = tmp_file.name
        
        # Redirect stdout to the temporary file while calling print_metrics_report
        with open(output_path, "w") as f:
            with contextlib.redirect_stdout(f):
                print_metrics_report(metrics)
    
    # Log the temporary file as an MLflow artifact
    # mlflow.log_artifact(filename, artifact_path="reports")
    # print(f"Metrics report logged as MLflow artifact: {filename}")
    return output_path



# def flatten_metrics(metrics_hierarchical: dict) -> dict:
#     """
#     Transform a hierarchical metrics dictionary into a flat dictionary
#     compatible with mlflow.log_metrics, using slashes for hierarchy.
    
#     Example of output key: "all_raw/level_1/overall_accuracy"
#     """
#     flattened = {}
    
#     for metric_type, values in metrics_hierarchical.items():
#         # Add n_samples as a global metric for the type
#         if 'n_samples' in values:
#             flattened[f"{metric_type}/n_samples"] = values['n_samples']
        
#         for key, val in values.items():
#             if key == 'n_samples':
#                 continue
#             # Every other key becomes <metric_type>/<key>
#             flattened[f"{metric_type}/{key}"] = val
    
#     return flattened

def flatten_metrics(
    metrics_hierarchical: dict, 
    include_product_types: bool = True,
    product_type_prefix: str = "product_type"
) -> dict:
    """
    Transform a hierarchical metrics dictionary into a flat dictionary
    compatible with mlflow.log_metrics, using slashes for hierarchy.
    
    Args:
        metrics_hierarchical: Dictionary returned by compute_hierarchical_metrics
        include_product_types: If True, include metrics for each product type
        product_type_prefix: Prefix for product type metrics (default: "product_type")
    
    Example of output keys:
        - "overall/all_raw/n_samples"
        - "overall/all_raw/level_1"
        - "overall/parsed_and_codable/level_1_retrieval_accuracy"
        - "product_type/01/all_raw/level_1"
        - "product_type/01/parsed_and_codable/level_3_generation_accuracy_when_retrieved"
    
    Returns:
        Flat dictionary with slash-separated keys
    """
    flattened = {}
    
    # ========== OVERALL METRICS ==========
    if 'overall' in metrics_hierarchical:
        for metric_type, values in metrics_hierarchical['overall'].items():
            # Add n_samples
            if 'n_samples' in values:
                flattened[f"overall/{metric_type}/n_samples"] = values['n_samples']
            
            # Add all other metrics
            for key, val in values.items():
                if key == 'n_samples':
                    continue
                flattened[f"overall/{metric_type}/{key}"] = val
    
    # ========== PRODUCT TYPE METRICS ==========
    if include_product_types and 'by_product_type' in metrics_hierarchical:
        for product_type, product_metrics in metrics_hierarchical['by_product_type'].items():
            for metric_type, values in product_metrics.items():
                # Add n_samples
                if 'n_samples' in values:
                    flattened[f"{product_type_prefix}/{product_type}/{metric_type}/n_samples"] = values['n_samples']
                
                # Add all other metrics
                for key, val in values.items():
                    if key == 'n_samples':
                        continue
                    # Only add non-None values
                    if val is not None:
                        flattened[f"{product_type_prefix}/{product_type}/{metric_type}/{key}"] = val
    
    return flattened