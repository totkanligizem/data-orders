import pandas as pd
import numpy as np

from olist.data import Olist


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance (km).
    lat/lon: pandas Series veya numpy array olabilir.
    """
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    R = 6371.0  # km
    return R * c


class Order:
    def __init__(self):
        self.data = Olist().get_data()

    def get_wait_time(self):
        orders = self.data["orders"].copy()

        # delivered filtreliyorum (testin beklediği satır sayısı buradan geliyor)
        orders = orders[orders["order_status"] == "delivered"].copy()

        # tarih kolonlarını datetime'a çeviriyorum
        date_cols = [
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
        for col in date_cols:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")

        # gün cinsinden ondalıklı farklar (dt.days kullanmıyorum, aşağı yuvarlar)
        wait_time = (
            orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]
        ) / np.timedelta64(1, "D")

        expected_wait_time = (
            orders["order_estimated_delivery_date"] - orders["order_purchase_timestamp"]
        ) / np.timedelta64(1, "D")

        delay_vs_expected = (
            orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
        ) / np.timedelta64(1, "D")

        # erken teslim negatif çıkar -> gecikme yoksa 0'a kırpıyorum
        delay_vs_expected = delay_vs_expected.clip(lower=0)

        result = pd.DataFrame(
            {
                "order_id": orders["order_id"],
                "wait_time": wait_time,
                "expected_wait_time": expected_wait_time,
                "delay_vs_expected": delay_vs_expected,
                "order_status": orders["order_status"],
            }
        )

        return result.reset_index(drop=True)

    def get_review_score(self):
        reviews = self.data["order_reviews"].copy()

        df = reviews[["order_id", "review_score"]].copy()
        df["dim_is_five_star"] = (df["review_score"] == 5).astype(int)
        df["dim_is_one_star"] = (df["review_score"] == 1).astype(int)

        df = df[["order_id", "dim_is_five_star", "dim_is_one_star", "review_score"]]
        return df.reset_index(drop=True)

    def get_number_items(self):
        items = self.data["order_items"].copy()
        df = items.groupby("order_id").size().reset_index(name="number_of_items")
        return df

    def get_number_sellers(self):
        items = self.data["order_items"].copy()
        df = (
            items.groupby("order_id")["seller_id"]
            .nunique()
            .reset_index(name="number_of_sellers")
        )
        return df

    def get_price_and_freight(self):
        items = self.data["order_items"].copy()
        df = (
            items.groupby("order_id")[["price", "freight_value"]]
            .sum()
            .reset_index()
        )
        return df

    def get_distance_seller_customer(self):
        orders = self.data["orders"].copy()
        customers = self.data["customers"].copy()
        sellers = self.data["sellers"].copy()
        items = self.data["order_items"].copy()
        geo = self.data["geolocation"].copy()

        # zip prefix -> tek koordinat (ortalama)
        geo_mean = (
            geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
            .mean()
            .reset_index()
        )

        # müşteri koordinatları
        customers_geo = (
            customers.merge(
                geo_mean,
                left_on="customer_zip_code_prefix",
                right_on="geolocation_zip_code_prefix",
                how="left",
            )[["customer_id", "geolocation_lat", "geolocation_lng"]]
            .rename(columns={"geolocation_lat": "cust_lat", "geolocation_lng": "cust_lng"})
        )

        # satıcı koordinatları
        sellers_geo = (
            sellers.merge(
                geo_mean,
                left_on="seller_zip_code_prefix",
                right_on="geolocation_zip_code_prefix",
                how="left",
            )[["seller_id", "geolocation_lat", "geolocation_lng"]]
            .rename(columns={"geolocation_lat": "sell_lat", "geolocation_lng": "sell_lng"})
        )

        # order_items içine customer_id'yi order_id üzerinden ekliyorum
        items_with_customer = items.merge(
            orders[["order_id", "customer_id"]],
            on="order_id",
            how="left",
        )

        merged = (
            items_with_customer
            .merge(customers_geo, on="customer_id", how="left")
            .merge(sellers_geo, on="seller_id", how="left")
        )

        # notebook'un/plot'un beklediği kolon adı bu
        merged["distance_seller_customer"] = haversine_distance(
            merged["cust_lat"],
            merged["cust_lng"],
            merged["sell_lat"],
            merged["sell_lng"],
        )

        df = (
            merged.groupby("order_id")["distance_seller_customer"]
            .mean()
            .reset_index()
        )

        return df

    def get_training_data(self, with_distance_seller_customer=False):
        df = self.get_wait_time()

        df = df.merge(self.get_review_score(), on="order_id", how="left")
        df = df.merge(self.get_number_items(), on="order_id", how="left")
        df = df.merge(self.get_number_sellers(), on="order_id", how="left")
        df = df.merge(self.get_price_and_freight(), on="order_id", how="left")

        if with_distance_seller_customer:
            df = df.merge(self.get_distance_seller_customer(), on="order_id", how="left")

        df = df.dropna().reset_index(drop=True)
        return df
        