from azure.storage.blob import BlobServiceClient
import os

class BlobStorageClient:
    def __init__(self, connection_string: str, container_name: str):
        self.connection_string = connection_string
        self.container_name = container_name
        
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string
        )
        self.container_client = self.blob_service_client.get_container_client(
            self.container_name
        )

    def list_blobs(self):
        """
        Returns list of blob paths inside the container.
        Example: ["topic1/file1.pdf", "topic1/file2.txt", "topic2/img1.png"]
        """
        blobs = []
        for blob in self.container_client.list_blobs():
            blobs.append(blob.name)
        
        return blobs

    def download_blob(self, blob_path: str):
        """
        Downloads raw bytes of a specific blob.
        Returns: file_bytes, file_type
        """
        blob_client = self.container_client.get_blob_client(blob_path)
        stream = blob_client.download_blob()
        file_bytes = stream.readall()

        # detect file type from extension
        ext = os.path.splitext(blob_path)[1].lower()
        file_type = ext.replace(".", "")

        return file_bytes, file_type

    def get_blob_url(self, blob_path: str) -> str:
        """
        Returns the URL for a blob given its path within the container.
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            return blob_client.url
        except Exception:
            # Fallback to constructing manually if blob_client doesn't provide url
            account_name = getattr(self.blob_service_client, "account_name", None)
            if account_name:
                return f"https://{account_name}.blob.core.windows.net/{self.container_name}/{blob_path}"
            raise

