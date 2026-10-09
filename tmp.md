    def test_sidecar_startup_logged(self):
        # SGLang's Python logging goes to stderr while Rust tracing and
        # uvicorn access logs go to stdout, so scan both streams.
        log_content = ""
        for log_file in (self.out_log_file, self.err_log_file):
            log_file.seek(0)
            log_content += log_file.read()
        self.assertIn(f"Sidecar module {SIDECAR_MODULE_NAME} started pid=", log_content)
