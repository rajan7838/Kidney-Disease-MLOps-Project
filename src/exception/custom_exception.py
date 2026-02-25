"""This is a custom exception class that captures detailed error information like file name 
and line number using sys.exc_info() and returns a formatted error message 
for better debugging in production projects."""



import sys

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        _, _, exc_tb = error_detail.exc_info()
        
        self.error_message = f"""
        File: {exc_tb.tb_frame.f_code.co_filename}
        Line: {exc_tb.tb_lineno}
        Error: {str(error_message)}
        """

    def __str__(self):
        return self.error_message



