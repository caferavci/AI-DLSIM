import React, { useMemo } from "react";
import { MapContainer, Polyline, TileLayer } from "react-leaflet";
import "leaflet/dist/leaflet.css";

function speedColor(speed) {
  if (speed == null) return "#888";
  if (speed >= 35) return "#2E8B57";
  if (speed >= 20) return "#E69F00";
  return "#CC3333";
}

export default function NetworkMap({ links, routePoints, mapKey }) {
  const center = useMemo(() => {
    const first = links?.find((l) => l.geometry?.length > 0);
    if (!first) return [42.443, -76.501];
    return first.geometry[0];
  }, [links]);

  const hasRoute = (routePoints || []).length > 1;

  return (
    <div className="map-panel">
      <MapContainer
        key={mapKey}
        center={center}
        zoom={13}
        scrollWheelZoom
        style={{ height: "520px", width: "100%" }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {links?.map((link) =>
          link.geometry?.length > 1 ? (
            <Polyline
              key={`l-${link.link_id}`}
              positions={link.geometry}
              pathOptions={{ color: speedColor(link.speed), weight: 2, opacity: 0.7 }}
            />
          ) : null
        )}
        {routePoints?.length > 1 ? (
          <Polyline
            positions={routePoints}
            pathOptions={{ color: "#2060ff", weight: 4, opacity: 0.95 }}
          />
        ) : null}
      </MapContainer>
      {!hasRoute ? <div className="map-note">No route output available for this engine.</div> : null}
    </div>
  );
}

