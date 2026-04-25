use crate::app_services::ctrader_openapi::{
    ProtoHeartbeatEvent, ProtoMessage, ProtoOAAccountAuthReq, ProtoOAApplicationAuthReq,
    ProtoOAPayloadType, ProtoPayloadType,
};
use protobuf::prelude::*;
use protobuf::{Message, Parse, Serialize};
use anyhow::{Result, Context};

pub fn build_proto_message<M: Message + Serialize>(payload_type: u32, message: &M, client_msg_id: Option<String>) -> Result<Vec<u8>> {
    let mut envelope = ProtoMessage::new();
    let mut mut_envelope = envelope.as_mut();
    mut_envelope.set_payloadType(payload_type);
    mut_envelope.set_payload(message.serialize()?);
    if let Some(id) = client_msg_id {
        mut_envelope.set_clientMsgId(id);
    }
    envelope.serialize().context("failed to serialize ProtoMessage envelope")
}

pub fn build_app_auth_req(client_id: &str, client_secret: &str, client_msg_id: Option<String>) -> Result<Vec<u8>> {
    let mut req = ProtoOAApplicationAuthReq::new();
    let mut mut_req = req.as_mut();
    mut_req.set_clientId(client_id.to_string());
    mut_req.set_clientSecret(client_secret.to_string());

    build_proto_message(i32::from(ProtoOAPayloadType::ProtoOaApplicationAuthReq) as u32, &req, client_msg_id)
}

pub fn build_account_auth_req(account_id: i64, access_token: &str, client_msg_id: Option<String>) -> Result<Vec<u8>> {
    let mut req = ProtoOAAccountAuthReq::new();
    let mut mut_req = req.as_mut();
    mut_req.set_ctidTraderAccountId(account_id);
    mut_req.set_accessToken(access_token.to_string());

    build_proto_message(i32::from(ProtoOAPayloadType::ProtoOaAccountAuthReq) as u32, &req, client_msg_id)
}

pub fn build_heartbeat() -> Result<Vec<u8>> {
    let req = ProtoHeartbeatEvent::new();
    build_proto_message(i32::from(ProtoPayloadType::HeartbeatEvent) as u32, &req, None)
}

pub fn parse_proto_message(data: &[u8]) -> Result<ProtoMessage> {
    ProtoMessage::parse(data).map_err(|e| anyhow::anyhow!("failed to parse ProtoMessage envelope: {:?}", e))
}
