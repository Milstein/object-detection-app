# Use Fedora-based Python 3.11 image from Quay
FROM quay.io/fedora/python-311

# Set working directory
WORKDIR /opt/app-root/src

# Switch to root to install system dependencies
USER root

# Install OpenGL-related dependencies
RUN dnf install -y \
    mesa-libGL \
    mesa-dri-drivers \
    libglvnd-glx \
    libglvnd-opengl \
    libX11 \
    && dnf clean all

# Switch back to non-root user for security
USER 1001

# Copy Python dependencies
COPY --chown=1001:0 requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -f requirements.txt && \
    chmod -R g+w /opt/app-root/lib/python3.11/site-packages

# Copy application files
# COPY --chown=1001:0 app.py coco.yaml remote_infer_grpc.py grpc_predict_v2_pb2_grpc.py grpc_predict_v2_pb2.py grpc_predict_v2.proto ./
COPY --chown=1001:0 app.py coco.yaml remote_infer_rest.py ./

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
