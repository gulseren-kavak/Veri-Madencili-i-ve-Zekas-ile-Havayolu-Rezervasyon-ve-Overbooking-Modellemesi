"""
Kapsamlı Veri Üretici — Overbooking Risk Yönetim Sistemi
=========================================================
Üretilen veri setleri:
  - data/flights.csv        : 15.000 uçuş kaydı (genişletilmiş özelliklerle)
  - data/airports.csv       : Havalimanı bilgileri
  - data/weather_data.csv   : Hava durumu verileri
  - data/events.csv         : Özel etkinlikler (maç, festival, tatil)
  - data/reservations.csv   : Rezervasyon ve kapasite bilgileri
"""
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. HAVAALANLARI
# ============================================================
airports_data = {
    "IATA_CODE": ["JFK","LAX","ORD","ATL","DFW","DEN","SFO","SEA","MIA","BOS",
                   "LAS","MCO","CLT","PHX","IAH","MSP","DTW","EWR","PHL","BWI"],
    "AIRPORT": [
        "John F. Kennedy Intl","Los Angeles Intl","O'Hare Intl","Hartsfield-Jackson Intl",
        "Dallas/Fort Worth Intl","Denver Intl","San Francisco Intl","Seattle-Tacoma Intl",
        "Miami Intl","Boston Logan Intl","McCarran Intl","Orlando Intl",
        "Charlotte Douglas Intl","Phoenix Sky Harbor","George Bush Intl",
        "Minneapolis-Saint Paul Intl","Detroit Metro Wayne","Newark Liberty Intl",
        "Philadelphia Intl","Baltimore-Washington Intl"
    ],
    "CITY": ["New York","Los Angeles","Chicago","Atlanta","Dallas","Denver",
             "San Francisco","Seattle","Miami","Boston","Las Vegas","Orlando",
             "Charlotte","Phoenix","Houston","Minneapolis","Detroit","Newark",
             "Philadelphia","Baltimore"],
    "STATE": ["NY","CA","IL","GA","TX","CO","CA","WA","FL","MA",
              "NV","FL","NC","AZ","TX","MN","MI","NJ","PA","MD"],
    "LATITUDE": [40.64,33.94,41.97,33.64,32.90,39.86,37.62,47.45,25.79,42.36,
                 36.08,28.43,35.21,33.43,29.98,44.88,42.21,40.69,39.87,39.17],
    "LONGITUDE": [-73.78,-118.41,-87.91,-84.43,-97.04,-104.67,-122.38,-122.31,
                  -80.29,-71.01,-115.15,-81.31,-80.94,-112.01,-95.34,-93.22,
                  -83.35,-74.17,-75.24,-76.67]
}
airports_df = pd.DataFrame(airports_data)
airports_df.to_csv(DATA_DIR / "airports.csv", index=False)
print(f"✅ airports.csv → {len(airports_df)} havalimanı")

# ============================================================
# 2. HAVAYOLLARI & UÇAK TİPLERİ
# ============================================================
airlines = {
    "AA": {"name": "American Airlines", "overbooking_rate": 0.06},
    "UA": {"name": "United Airlines",   "overbooking_rate": 0.07},
    "DL": {"name": "Delta Air Lines",   "overbooking_rate": 0.04},
    "WN": {"name": "Southwest Airlines","overbooking_rate": 0.08},
    "B6": {"name": "JetBlue Airways",   "overbooking_rate": 0.05},
    "AS": {"name": "Alaska Airlines",   "overbooking_rate": 0.03},
    "NK": {"name": "Spirit Airlines",   "overbooking_rate": 0.10},
    "F9": {"name": "Frontier Airlines", "overbooking_rate": 0.09},
    "HA": {"name": "Hawaiian Airlines",  "overbooking_rate": 0.02},
    "SY": {"name": "Sun Country",       "overbooking_rate": 0.07},
}

aircraft_types = {
    "B737": {"capacity": 160, "business_seats": 16},
    "B738": {"capacity": 189, "business_seats": 20},
    "A320": {"capacity": 180, "business_seats": 18},
    "A321": {"capacity": 220, "business_seats": 24},
    "B777": {"capacity": 350, "business_seats": 40},
    "B787": {"capacity": 290, "business_seats": 32},
    "A330": {"capacity": 280, "business_seats": 30},
    "E175": {"capacity": 80,  "business_seats": 12},
}

# ============================================================
# 3. ETKİNLİKLER
# ============================================================
events_list = []
base_date = datetime(2026, 1, 1)

event_templates = [
    # (event_name, city, month, day, duration_days, impact_factor)
    ("Super Bowl", "Phoenix", 2, 8, 3, 0.95),
    ("NBA All-Star Game", "Atlanta", 2, 15, 2, 0.80),
    ("Mardi Gras", "New York", 2, 25, 5, 0.70),
    ("Spring Break Peak", "Miami", 3, 15, 14, 0.85),
    ("Masters Golf Tournament", "Atlanta", 4, 10, 4, 0.75),
    ("Boston Marathon", "Boston", 4, 20, 2, 0.65),
    ("Kentucky Derby", "Charlotte", 5, 3, 3, 0.60),
    ("Memorial Day Weekend", "ALL", 5, 24, 4, 0.80),
    ("Summer Travel Peak", "ALL", 6, 15, 60, 0.50),
    ("Independence Day", "ALL", 7, 2, 5, 0.90),
    ("Comic-Con", "San Francisco", 7, 20, 5, 0.70),
    ("Music Festival – Lollapalooza", "Chicago", 7, 31, 4, 0.75),
    ("US Open Tennis", "New York", 8, 25, 14, 0.80),
    ("Labor Day Weekend", "ALL", 9, 1, 4, 0.85),
    ("Halloween Weekend", "Las Vegas", 10, 29, 4, 0.70),
    ("Thanksgiving", "ALL", 11, 25, 7, 0.95),
    ("Christmas & New Year", "ALL", 12, 20, 15, 0.92),
    ("Barcelona FC vs Real Madrid (El Clasico TV Event)", "New York", 10, 26, 2, 0.60),
    ("World Cup Qualifier", "Houston", 3, 25, 2, 0.65),
    ("Street Food Festival", "Los Angeles", 4, 5, 3, 0.55),
    ("Tech Summit CES", "Las Vegas", 1, 7, 5, 0.85),
    ("Art Basel", "Miami", 12, 1, 5, 0.80),
]

for name, city, month, day, dur, impact in event_templates:
    try:
        start = datetime(2026, month, day)
    except ValueError:
        start = datetime(2026, month, min(day, 28))
    for d in range(dur):
        dt = start + timedelta(days=d)
        events_list.append({
            "DATE": dt.strftime("%Y-%m-%d"),
            "EVENT_NAME": name,
            "CITY": city,
            "IMPACT_FACTOR": round(impact + np.random.uniform(-0.05, 0.05), 3),
            "EVENT_TYPE": "sport" if any(w in name.lower() for w in ["bowl","nba","golf","marathon","derby","open","world cup","barcelona","clasico"]) else
                          "holiday" if any(w in name.lower() for w in ["memorial","independence","thanksgiving","christmas","labor"]) else
                          "festival" if any(w in name.lower() for w in ["festival","mardi","comic","music","art","food"]) else "conference",
        })

events_df = pd.DataFrame(events_list)
events_df.to_csv(DATA_DIR / "events.csv", index=False)
print(f"✅ events.csv → {len(events_df)} etkinlik-gün kaydı")

# ============================================================
# 4. HAVA DURUMU VERİSİ (365 gün × 20 havalimanı)
# ============================================================
weather_rows = []
for ap_code in airports_data["IATA_CODE"]:
    idx = airports_data["IATA_CODE"].index(ap_code)
    lat = airports_data["LATITUDE"][idx]
    for day_offset in range(365):
        dt = base_date + timedelta(days=day_offset)
        month = dt.month
        # Seasonal temperature
        if lat > 40:  # Northern
            base_temp = 0 + 25 * np.sin(np.pi * (month - 1) / 6)
        else:  # Southern / warmer
            base_temp = 15 + 15 * np.sin(np.pi * (month - 1) / 6)
        temp = round(base_temp + np.random.normal(0, 5), 1)
        # Weather conditions
        precip = round(max(0, np.random.exponential(3)), 1)
        wind = round(max(0, np.random.normal(15, 8)), 1)
        visibility = round(max(1, min(10, 10 - precip * 0.5 + np.random.normal(0, 1))), 1)
        condition_roll = random.random()
        if precip > 10:
            condition = "Storm"
        elif precip > 5:
            condition = "Rain"
        elif precip > 2:
            condition = "Cloudy"
        elif visibility < 5:
            condition = "Fog"
        elif temp < -5:
            condition = "Snow"
        else:
            condition = "Clear"

        weather_rows.append({
            "DATE": dt.strftime("%Y-%m-%d"),
            "AIRPORT": ap_code,
            "TEMPERATURE_C": temp,
            "PRECIPITATION_MM": precip,
            "WIND_SPEED_KMH": wind,
            "VISIBILITY_KM": visibility,
            "CONDITION": condition,
            "DELAY_RISK_FACTOR": round(
                0.1 * (precip > 5) + 0.15 * (wind > 30) + 0.1 * (visibility < 5) +
                0.05 * (temp < -5) + 0.2 * (condition == "Storm") + np.random.uniform(0, 0.05), 3
            )
        })

weather_df = pd.DataFrame(weather_rows)
weather_df.to_csv(DATA_DIR / "weather_data.csv", index=False)
print(f"✅ weather_data.csv → {len(weather_df)} kayıt")

# ============================================================
# 5. UÇUŞLAR + REZERVASYONLAR (15.000 uçuş)
# ============================================================
NUM_FLIGHTS = 15000

# Prepare event lookup: date+city → impact
event_lookup = {}
for _, row in events_df.iterrows():
    key = (row["DATE"], row["CITY"])
    event_lookup[key] = max(event_lookup.get(key, 0), row["IMPACT_FACTOR"])
    # ALL‑city events
    if row["CITY"] == "ALL":
        for city in airports_data["CITY"]:
            key2 = (row["DATE"], city)
            event_lookup[key2] = max(event_lookup.get(key2, 0), row["IMPACT_FACTOR"])

# Weather lookup
weather_lookup = {}
for _, row in weather_df.iterrows():
    weather_lookup[(row["DATE"], row["AIRPORT"])] = row

iata_codes = airports_data["IATA_CODE"]
city_of = dict(zip(airports_data["IATA_CODE"], airports_data["CITY"]))
airline_codes = list(airlines.keys())

flights = []
reservations = []

for i in range(NUM_FLIGHTS):
    # Random date in 2026
    day_offset = random.randint(0, 364)
    dt = base_date + timedelta(days=day_offset)
    date_str = dt.strftime("%Y-%m-%d")
    month = dt.month
    day_of_week = dt.weekday()  # 0=Mon

    # Airline
    al = random.choice(airline_codes)
    al_info = airlines[al]

    # Route
    origin = random.choice(iata_codes)
    dest = random.choice([c for c in iata_codes if c != origin])

    # Aircraft
    ac_type = random.choice(list(aircraft_types.keys()))
    ac_info = aircraft_types[ac_type]
    capacity = ac_info["capacity"]
    business_seats = ac_info["business_seats"]

    # Departure time
    hour = random.choices(range(6, 23), weights=[2,3,5,6,7,7,8,6,5,5,6,7,6,4,3,2,1])[0]
    minute = random.choice([0, 15, 30, 45])
    sched_dep = hour * 100 + minute

    # Distance
    lat1 = airports_data["LATITUDE"][iata_codes.index(origin)]
    lat2 = airports_data["LATITUDE"][iata_codes.index(dest)]
    lon1 = airports_data["LONGITUDE"][iata_codes.index(origin)]
    lon2 = airports_data["LONGITUDE"][iata_codes.index(dest)]
    distance = int(((lat2 - lat1)**2 + (lon2 - lon1)**2)**0.5 * 69)  # rough miles

    # ---- OVERBOOKING LOGIC ----
    base_ob_rate = al_info["overbooking_rate"]

    # Event impact
    origin_city = city_of[origin]
    dest_city = city_of[dest]
    event_impact_origin = event_lookup.get((date_str, origin_city), 0)
    event_impact_dest = event_lookup.get((date_str, dest_city), 0)
    event_impact = max(event_impact_origin, event_impact_dest)
    has_event = 1 if event_impact > 0 else 0

    # Weather impact
    weather_origin = weather_lookup.get((date_str, origin), {})
    weather_dest = weather_lookup.get((date_str, dest), {})
    weather_delay_risk = 0
    if isinstance(weather_origin, dict):
        weather_delay_risk = weather_origin.get("DELAY_RISK_FACTOR", 0)
    elif hasattr(weather_origin, "DELAY_RISK_FACTOR"):
        weather_delay_risk = weather_origin.DELAY_RISK_FACTOR

    # Seasonal effect
    seasonal_factor = 0.3 * np.sin(np.pi * (month - 3) / 6) + 0.1  # peaks in summer
    seasonal_factor = max(0, seasonal_factor)

    # Day of week (Fri/Sun busier)
    dow_factor = 0.15 if day_of_week in [4, 6] else 0.05 if day_of_week in [0, 3] else 0

    # Compute booking ratio
    # Higher event impact → more bookings → higher overbooking
    extra_demand = (event_impact * 0.3 + seasonal_factor * 0.2 + dow_factor +
                    base_ob_rate + np.random.normal(0, 0.05))
    booking_ratio = min(1.25, max(0.6, 0.85 + extra_demand))
    tickets_sold = int(capacity * booking_ratio)

    # Expected no‑shows
    no_show_rate = max(0.01, np.random.normal(0.05, 0.02))
    expected_passengers = int(tickets_sold * (1 - no_show_rate))

    # OVERBOOKING label: passengers > capacity
    is_overbooked = 1 if expected_passengers > capacity else 0

    # Delay (weather + event‑driven)
    delay_minutes = max(0, int(
        weather_delay_risk * 120 +
        (event_impact * 30 if event_impact > 0.7 else 0) +
        np.random.normal(0, 15)
    ))

    flight_number = random.randint(100, 9999)

    flights.append({
        "FLIGHT_ID": f"{al}{flight_number}_{date_str}",
        "DATE": date_str,
        "MONTH": month,
        "DAY_OF_WEEK": day_of_week,
        "AIRLINE": al,
        "FLIGHT_NUMBER": flight_number,
        "ORIGIN_AIRPORT": origin,
        "DESTINATION_AIRPORT": dest,
        "AIRCRAFT_TYPE": ac_type,
        "SCHEDULED_DEPARTURE": sched_dep,
        "DISTANCE": distance,
        "CAPACITY": capacity,
        "TICKETS_SOLD": tickets_sold,
        "EXPECTED_PASSENGERS": expected_passengers,
        "BOOKING_RATIO": round(booking_ratio, 4),
        "NO_SHOW_RATE": round(no_show_rate, 4),
        "HAS_EVENT": has_event,
        "EVENT_IMPACT": round(event_impact, 3),
        "WEATHER_DELAY_RISK": round(weather_delay_risk, 3),
        "SEASONAL_FACTOR": round(seasonal_factor, 3),
        "DAY_OF_WEEK_FACTOR": round(dow_factor, 3),
        "DELAY_MINUTES": delay_minutes,
        "IS_OVERBOOKED": is_overbooked,
    })

    # Reservation record
    reservations.append({
        "FLIGHT_ID": flights[-1]["FLIGHT_ID"],
        "DATE": date_str,
        "AIRLINE": al,
        "FLIGHT_NUMBER": flight_number,
        "ORIGIN_AIRPORT": origin,
        "DESTINATION_AIRPORT": dest,
        "CAPACITY": capacity,
        "BUSINESS_SEATS": business_seats,
        "ECONOMY_SEATS": capacity - business_seats,
        "TICKETS_SOLD": tickets_sold,
        "BUSINESS_SOLD": min(business_seats, int(tickets_sold * random.uniform(0.6, 1.0) * business_seats / capacity)),
        "ECONOMY_SOLD": tickets_sold - min(business_seats, int(tickets_sold * random.uniform(0.6, 1.0) * business_seats / capacity)),
        "BOOKING_RATIO": round(booking_ratio, 4),
        "WAITLIST_COUNT": max(0, tickets_sold - capacity),
        "NO_SHOW_ESTIMATE": round(no_show_rate, 4),
        "EXPECTED_PASSENGERS": expected_passengers,
        "OVERBOOKED_COUNT": max(0, expected_passengers - capacity),
        "UPGRADE_COST_USD": round(max(0, expected_passengers - capacity) * random.uniform(200, 600), 2) if expected_passengers > capacity else 0,
        "HOTEL_COST_USD": round(max(0, expected_passengers - capacity - business_seats + min(business_seats, int(tickets_sold * random.uniform(0.6, 1.0) * business_seats / capacity))) * random.uniform(150, 400), 2) if expected_passengers > capacity else 0,
        "COMPENSATION_COST_USD": round(max(0, expected_passengers - capacity) * random.uniform(300, 1200), 2) if expected_passengers > capacity else 0,
    })

flights_df = pd.DataFrame(flights)
reservations_df = pd.DataFrame(reservations)

flights_df.to_csv(DATA_DIR / "flights.csv", index=False)
reservations_df.to_csv(DATA_DIR / "reservations.csv", index=False)

ob_count = flights_df["IS_OVERBOOKED"].sum()
ob_pct = ob_count / len(flights_df) * 100
print(f"✅ flights.csv → {len(flights_df)} uçuş ({ob_count} overbooked = %{ob_pct:.1f})")
print(f"✅ reservations.csv → {len(reservations_df)} rezervasyon")

total_upgrade = reservations_df["UPGRADE_COST_USD"].sum()
total_hotel = reservations_df["HOTEL_COST_USD"].sum()
total_comp = reservations_df["COMPENSATION_COST_USD"].sum()
print(f"\n💰 Toplam Maliyet Tahmini:")
print(f"   Business yükseltme: ${total_upgrade:,.0f}")
print(f"   Otel konaklatma:    ${total_hotel:,.0f}")
print(f"   Tazminat:           ${total_comp:,.0f}")
print(f"   TOPLAM:             ${total_upgrade + total_hotel + total_comp:,.0f}")
