# Dockerfile
FROM registry.access.redhat.com/ubi9/python-311

# Set working directory
WORKDIR /opt/app-root/src

USER 1001

COPY --chown=1001:0 requirements.txt ./

# Copy requirements and install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -f requirements.txt && \
    # Fix permissions to support pip in Openshift environments \
    chmod -R g+w /opt/app-root/lib/python3.11/site-packages && \
    fix-permissions /opt/app-root -P

# Copy application files
COPY --chown=1001:0 app.py coco.yaml remote_infer_grpc.py ./
# COPY --chown=1001:0 assets/ ./assets/

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
