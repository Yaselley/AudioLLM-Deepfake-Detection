"""
OpenSMILE Feature Text Generator (Selected Features)
Extracts specific OpenSMILE features based on indices.
"""
from typing import Dict, List, Optional
import numpy as np
from .opensmile_features import OpenSMILETextExtractor

class OpenSMILESelectedExtractor(OpenSMILETextExtractor):
    """
    Extracts selected OpenSMILE features based on provided indices.
    """
    
    def __init__(self, selected_indices: Optional[List[int]] = None):
        """
        Initialize the extractor with optional selected indices.
        
        Args:
            selected_indices: List of integer indices of features to keep.
                              If None, all features are kept.
        """
        super().__init__()
        self.selected_indices = selected_indices
        
        # If indices are provided, filter the feature names
        if self.selected_indices is not None:
            # Validate and select feature names
            self.selected_feature_names = []
            for idx in self.selected_indices:
                if 0 <= idx < len(self.feature_names):
                    self.selected_feature_names.append(self.feature_names[idx])
                else:
                    print(f"Warning: Index {idx} out of bounds (0-{len(self.feature_names)-1}), skipping.")
            
            # Update the number of features to reflect the selection
            self.num_features = len(self.selected_feature_names)
            print(f"Selected {self.num_features} features based on provided indices.")
        else:
            self.selected_feature_names = self.feature_names

    def features_to_text(self, features: Dict[str, float], precision: int = 10) -> str:
        """
        Convert selected features to textual description.
        
        Args:
            features: Dictionary of feature name -> value
            precision: Number of decimal places for feature values
        
        Returns:
            Text description of selected features
        """
        lines = [
            f"Here are {self.num_features} speech features of the current speaker's sentence.",
            "The features are:"
        ]
        
        count = 0
        total_selected = len(self.selected_feature_names)
        
        for feature_name in self.selected_feature_names:
            if feature_name in features:
                value = features[feature_name]
                count += 1
                
                # Get human-readable name or use original
                readable_name = self.FEATURE_DESCRIPTIONS.get(feature_name, feature_name)
                formatted_value = f"{value:.{precision}f}"
                
                # Add feature description
                if count < total_selected:
                    lines.append(f"{readable_name}: {formatted_value},")
                else:
                    lines.append(f"{readable_name}: {formatted_value}.")
            else:
                # Should not happen if features come from the same extractor
                pass
        
        return " ".join(lines)

# Example usage
if __name__ == "__main__":
    # Initialize extractor with specific indices
    # Example: Select features at indices 2, 5, and 10
    selected_indices = [1, 18, 19, 20, 21, 39, 45, 51, 58, 60, 67, 77, 78, 82, 85, 86]
    extractor = OpenSMILESelectedExtractor(selected_indices=selected_indices)
    
    # Process single file (using a dummy path or one from the system if available)
    # Since we don't have a guaranteed file, we can test with the array method which uses dummy signal in __init__? 
    # No, __init__ uses dummy to get names.
    
    # We can create a dummy array to test
    dummy_audio = np.random.uniform(-1, 1, 16000)
    text_description = extractor.process_array(dummy_audio, sample_rate=16000, precision=5)
    print("\nGenerated Text:")
    print(text_description)
