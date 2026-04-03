import base64
from github import Github, InputGitTreeElement
import os

# Configuration
GITHUB_TOKEN = "YOUR_GITHUB_TOKEN"
REPO_NAME = "your_username/your_repository"
FILE_PATH_LOCAL = "your_local_binary_file.bin"    # immune_model.zip
FILE_PATH_GITHUB = "path/in/github/repository.bin"
COMMIT_MESSAGE = "Upload model"
BRANCH_NAME = "main" # Or your target branch name


def upload_binary_file(token, repo_name, local_path, github_path, commit_message, branch_name):
    """
    Uploads a binary file to a GitHub repository using PyGithub.
    """
    # Authenticate with GitHub
    g = Github(token)
    repo = g.get_repo(repo_name)

    # Read the binary file and base64 encode it
    with open(local_path, 'rb') as file:
        binary_content = file.read()
        encoded_content = base64.b64encode(binary_content).decode("utf-8")

    # Check if the file already exists to decide between create or update
    try:
        # Get the existing file's SHA if it exists (needed for updates)
        contents = repo.get_contents(github_path, ref=branch_name)
        # Update the file
        repo.update_file(
            contents.path,
            commit_message,
            encoded_content,
            contents.sha,
            branch=branch_name
        )
        print(f"File '{github_path}' updated successfully.")

    except Exception:
        # File does not exist, create it
        repo.create_file(
            github_path,
            commit_message,
            encoded_content,
            branch=branch_name
        )
        print(f"File '{github_path}' created successfully.")

if __name__ == "__main__":
    upload_binary_file(
        GITHUB_TOKEN,
        REPO_NAME,
        FILE_PATH_LOCAL,
        FILE_PATH_GITHUB,
        COMMIT_MESSAGE,
        BRANCH_NAME
    )

