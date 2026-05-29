# Contributing Guidelines

## Code Style

- Follow PEP 8 conventions
- Use meaningful variable and function names
- Add docstrings to functions explaining their purpose
- Include type hints where applicable

## Workflow

1. Create a new branch for your feature
2. Make your changes
3. Ensure no data files are committed
4. Test your changes
5. Submit a pull request with a clear description

## Data Handling

**IMPORTANT**: This repository should NOT contain any data files. 
- Keep `.gitignore` up to date with data directories
- Document data requirements in README
- Use relative paths for data access

## Testing

Before submitting changes:
- Test your code with sample data
- Ensure backward compatibility
- Document any new dependencies

## Commit Messages

Use clear, descriptive commit messages:
```
Add feature description

Detailed explanation of changes if needed.
```

## Code Review

All submissions will be reviewed for:
- Code quality and style
- Adherence to guidelines
- No accidental data commits
- Documentation updates
