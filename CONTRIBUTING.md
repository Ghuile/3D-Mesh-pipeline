# Contributing Guidelines

Welcome! This is a research project focused on 3D mesh processing and biometric analysis. We appreciate contributions that improve the codebase, documentation, and overall project quality.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a virtual environment
4. Install development dependencies: `pip install -r requirements.txt`
5. Create a feature branch from `main`

## Code Style

- **PEP 8**: Follow Python Enhancement Proposal 8 conventions
- **Naming**: Use meaningful, descriptive names for variables and functions
- **Docstrings**: Add docstrings to all functions explaining purpose, parameters, and return values
- **Type Hints**: Include type hints for function signatures where applicable
- **Comments**: Comment complex logic, but keep comments concise and accurate
- **Line Length**: Aim for maximum 100 characters per line

### Example Function

```python
def extract_biometric_attributes(mesh_path: str, output_dir: str) -> Dict[str, float]:
    """
    Extract biometric attributes from a 3D mesh file.
    
    Args:
        mesh_path: Path to the input mesh file (FBX, OBJ, etc.)
        output_dir: Directory to save extracted features
        
    Returns:
        Dictionary containing extracted attributes (height, fat_percentage, etc.)
        
    Raises:
        FileNotFoundError: If mesh_path does not exist
        ValueError: If mesh format is not supported
    """
    # Implementation here
    pass
```

## Development Workflow

1. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** while following code style guidelines

3. **Test your code**:
   ```bash
   # Test with sample data
   python your_script.py --test
   
   # Verify imports work
   python -c "from your_module import YourClass"
   ```

4. **Update documentation**:
   - Add docstrings to new functions
   - Update README.md if behavior changes
   - Document any new dependencies

5. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: brief description"
   ```

6. **Push and open a Pull Request**

## Data Handling

⚠️ **IMPORTANT**: This repository should **NOT** contain any data files.

- **No data commits**: Ensure `.gitignore` includes all data directories (`data_generation/`, `data_parsed/`, `data_sanitized/`)
- **Document requirements**: Describe expected data formats in script docstrings
- **Use relative paths**: For portability, use relative paths for data access
- **Validate paths**: Check that files exist before processing
- **Add .gitignore entries**: If you create new data directories, add them to `.gitignore`

### Data Directory Structure

```
data_generation/    # Synthetic data generation outputs (not committed)
data_parsed/        # Preprocessed mesh data (not committed)
data_sanitized/     # Sanitized/cleaned data (not committed)
```

## Testing Requirements

Before submitting a pull request:

- ✅ Test code with sample data
- ✅ Verify backward compatibility with existing code
- ✅ Ensure no hardcoded paths or credentials
- ✅ Check that new dependencies are documented in requirements.txt
- ✅ Validate error handling for edge cases
- ✅ Test on both GPU and CPU if applicable

### Running Tests

```bash
# Test individual script
python script_name.py --help

# Verify imports
python -c "import module_name; print('OK')"

# Quick functionality test
python -m pytest tests/ -v  # if tests exist
```

## Commit Messages

Use clear, descriptive commit messages following this format:

```
Short description (50 chars max)

Longer explanation of the changes made, why they were needed,
and any important context. Wrap at 72 characters.

- List specific changes
- One change per bullet point
```

### Examples

✅ **Good**:
```
Add heatmap visualization for mesh attributes

Implement visualization of vertex-level attributes on 3D meshes.
Uses matplotlib for rendering and color mapping. Includes support
for custom color schemes and output formats.

- Add metahuman_generate_heatmaps.py
- Support PNG and PDF exports
- Add colormap customization
```

❌ **Bad**:
```
fixes stuff
```

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows PEP 8 style guide
- [ ] All new functions have docstrings
- [ ] No data files are committed
- [ ] Changes maintain backward compatibility
- [ ] README/documentation is updated if needed
- [ ] New dependencies are documented

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Related Issues
Closes #(issue number)

## Testing
Describe how you tested the changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No data files committed
```

## Code Review

All submissions undergo review for:

1. **Code Quality**: Readability, efficiency, and maintainability
2. **Style Compliance**: Adherence to PEP 8 and project guidelines
3. **Data Safety**: No accidental data or credential commits
4. **Documentation**: Adequate docstrings and comments
5. **Testing**: Code tested with sample data
6. **Dependencies**: New packages properly documented

## Integration with External Tools

### Blender Integration
- Ensure `bpy` is properly installed: `blender --python -m pip install bpy`
- Test scripts in Blender context when modifying `sanitize_pipeline_*.py`
- Document any Blender version requirements

### Unreal Engine Integration
- Test Unreal Engine scripts in appropriate UE5 environment
- Document Python version compatibility
- Provide fallback for environments without UE5

## Performance Considerations

When contributing code that processes large meshes:

- Consider memory footprint (target: <6GB VRAM)
- Profile code for bottlenecks
- Document computational complexity
- Provide options for batch processing if applicable

## Documentation

### Adding Documentation

- Update README.md for new features
- Add docstrings following Google/NumPy style
- Include usage examples in script headers
- Document any configuration parameters

### Example Script Header

```python
"""
Module: metahuman_extract_latents.py
Purpose: Extract latent representations from trained 3D autoencoder

Usage:
    python metahuman_extract_latents.py --model path/to/model.pth --data data_parsed/

Author: Your Name
Date: 2024-XX-XX
"""
```

## Questions or Need Help?

- Check existing issues and PRs
- Review script docstrings for detailed information
- Open an issue with the `question` label

## License Compliance

By contributing, you agree that your contributions will be licensed under the MIT License and comply with all third-party licensing requirements (Unreal Engine EULA, Blender GPL, etc.).

Thank you for contributing! 🎉
