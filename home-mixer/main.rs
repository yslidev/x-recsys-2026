use std::time::Duration;

use clap::Parser;
use xai_dark_traffic::RejectDarkTrafficLayer;
use xai_home_mixer::dark_traffic_setup;
use xai_home_mixer::params;
use xai_home_mixer::{HomeMixerConfig, HomeMixerServer};
use xai_home_mixer_proto as pb;
use xai_x_rpc::grpc_client::TlsMode;
use xai_x_rpc::wily_lookup_service::ShardCoordinate;
use xai_x_service_builder::XServiceBuilder;

#[derive(Parser, Debug)]
#[command(about = "HomeMixer gRPC Server")]
struct Args {
    #[arg(long, default_value_t = 50051u16)]
    grpc_port: u16,
    #[arg(long, default_value_t = 9090u16)]
    metrics_port: u16,
    #[arg(long, default_value_t = -1)]
    shard_coordinate: i16,
    #[arg(long, default_value_t = 500)]
    shard_total_size: u16,
    #[arg(long, default_value = "atla")]
    datacenter: String,
    #[arg(long, default_value = "")]
    otel_endpoint: String,
}

fn parse_shard(args: &Args) -> Option<ShardCoordinate> {
    if args.shard_coordinate >= 0 {
        Some(ShardCoordinate {
            ordinal: args.shard_coordinate as u16,
            total_size: args.shard_total_size,
        })
    } else {
        None
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let shard_coordinate = parse_shard(&args);

    xai_stringcenter::init_from_file(params::STRINGCENTER_BUNDLE_PATH);

    XServiceBuilder::new("home-mixer")
        .grpc_port(args.grpc_port)
        .metrics_port(args.metrics_port)
        .datacenter(args.datacenter)
        .otel_endpoint(args.otel_endpoint)
        .with_featureswitches(params::FS_PATH, true)
        .with_decider(params::decider_path(), None)
        .with_tls(TlsMode::server_mtls_from_env()?)
        .with_max_connection_age(Duration::from_secs(300))
        .with_reflection(pb::FILE_DESCRIPTOR_SET)
        .with_layer(dark_traffic_setup::resolve_layer())
        .with_layer(RejectDarkTrafficLayer::from_env())
        .http_routes(xai_profiling::profiling_router())
        .run::<HomeMixerServer>(HomeMixerConfig { shard_coordinate })
        .await
}
