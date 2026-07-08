use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("unauthorized")]
    Unauthorized,

    #[error("unknown or already-resolved request_id")]
    UnknownRequest,

    #[error("request_id was not assigned to this connection")]
    NotAssignedToCaller,

    #[error("{0}")]
    Rejected(String),

    #[error("{0}")]
    Rpc(String),

    #[error(transparent)]
    Db(#[from] sqlx::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl AppError {
    /// Stratum-style `[code, message, null]` error triple.
    pub fn to_stratum_error(&self) -> serde_json::Value {
        let code = match self {
            AppError::Unauthorized => 24,
            AppError::UnknownRequest => 20,
            AppError::NotAssignedToCaller => 25,
            AppError::Rejected(_) => 20,
            AppError::Rpc(_) => 20,
            AppError::Db(_) | AppError::Other(_) => 20,
        };
        serde_json::json!([code, self.to_string(), null_value()])
    }
}

fn null_value() -> serde_json::Value {
    serde_json::Value::Null
}
