import streamlit as st
import pandas as pd
import joblib
from PIL import Image


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="EstateAI Royal",
    page_icon="👑",
    layout="wide"
)


# ==============================
# LOAD MODEL
# ==============================

model = joblib.load(
    "house_price_xgb_model.pkl"
)


# ==============================
# ROYAL CSS
# ==============================

st.markdown("""

<style>


.stApp{

background:
radial-gradient(
circle at top,
#5b4608,
#090909 45%,
#000000
);

color:white;

}


header{

visibility:hidden;

}


h1,h2,h3,h4,h5,p,label{

color:white !important;

}


/* SIDEBAR */

section[data-testid="stSidebar"]{

background:

linear-gradient(
180deg,
#000000,
#211600
);

border-right:

1px solid #d4af37;

}


/* HERO */

.hero{


background:

linear-gradient(
135deg,
rgba(212,175,55,.4),
rgba(0,0,0,.8)
);


padding:50px;

border-radius:40px;

text-align:center;

border:

1px solid #d4af37;


box-shadow:

0 0 70px rgba(212,175,55,.5);


transition:.5s;

}


.hero:hover{

transform:translateY(-10px);

box-shadow:

0 0 120px rgba(212,175,55,.9);

}



.hero h1{

font-size:75px;

font-weight:1000;

background:

linear-gradient(
90deg,
#fff5b7,
#d4af37
);

-webkit-background-clip:text;

color:transparent;

}



.hero p{

font-size:22px;

}


/* CARD */


.card{

background:

rgba(255,255,255,.08);


padding:35px;


border-radius:35px;


border:

1px solid rgba(212,175,55,.5);


box-shadow:

0 20px 60px black;


transition:.4s;


}


.card:hover{

transform:translateY(-8px);


box-shadow:

0 0 50px #d4af37;

}


/* BUTTON */


.stButton button{


height:65px;

width:100%;


border-radius:40px;


font-size:22px;


font-weight:bold;


background:

linear-gradient(
90deg,
#d4af37,
#ffe98a
);


color:black;


transition:.3s;


}


.stButton button:hover{

transform:scale(1.08);

box-shadow:

0 0 40px #d4af37;

}


/* PRICE */


.price-card{


background:

linear-gradient(
135deg,
#d4af37,
#8b6508
);


padding:50px;


border-radius:45px;


text-align:center;


box-shadow:

0 0 80px #d4af37;


}


.price-card h1{


font-size:70px;


color:black !important;


}



.badge{


background:

linear-gradient(
90deg,
#fff5b7,
#d4af37
);


padding:15px;


border-radius:50px;


color:black;


font-size:25px;


font-weight:bold;


text-align:center;

}



.metric-box{


background:

rgba(212,175,55,.15);


border:

1px solid #d4af37;


padding:20px;


border-radius:25px;


text-align:center;


transition:.3s;


}


.metric-box:hover{


transform:translateY(-10px);


box-shadow:

0 0 40px #d4af37;


}


</style>

""",
unsafe_allow_html=True
)



# ==============================
# SIDEBAR
# ==============================


with st.sidebar:


    st.title("👑 EstateAI")


    st.write(
"""
Luxury AI Real Estate
Valuation System


Machine Learning Model:

XGBoost Regression


Accuracy:

R² Score ≈ 66%

"""
)



# ==============================
# HERO
# ==============================


st.markdown(
"""

<div class="hero">

<h1>
👑 EstateAI
</h1>

<p>
Royal AI Powered House Valuation
</p>


<p>
Predict your property's true market value
</p>


</div>

<br>

""",

unsafe_allow_html=True
)



# ==============================
# IMAGE + INPUT
# ==============================


left,right = st.columns([1,1.5])



with left:


    st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
    )


    st.subheader(
        "🏰 Property Image"
    )


    image = st.file_uploader(
        "Upload House Image",
        type=[
            "png",
            "jpg",
            "jpeg"
        ]
    )


    if image:

        img = Image.open(image)

        st.image(
            img,
            width=350
        )


    else:

        st.info(
            "Upload property image"
        )



    st.markdown(
    "</div>",
    unsafe_allow_html=True
    )




with right:


    st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
    )


    st.subheader(
        "🏡 Property Information"
    )


    city = st.selectbox(
        "City",
        [
            "Ahmedabad",
            "Surat",
            "Mumbai",
            "Delhi",
            "Vadodara"
        ]
    )


    c1,c2 = st.columns(2)


    with c1:


        area = st.number_input(
            "Area (sq.ft)",
            500,
            10000,
            3000
        )


        bedrooms = st.number_input(
            "Bedrooms",
            1,
            10,
            3
        )


        bathrooms = st.number_input(
            "Bathrooms",
            1,
            10,
            2
        )


    with c2:


        stories = st.number_input(
            "Stories",
            1,
            5,
            2
        )


        parking = st.number_input(
            "Parking",
            0,
            5,
            1
        )


        furnishing = st.selectbox(
            "Furnishing",
            [
                "furnished",
                "semi-furnished",
                "unfurnished"
            ]
        )



    guestroom = st.selectbox(
        "Guest Room",
        ["yes","no"]
    )


    basement = st.selectbox(
        "Basement",
        ["yes","no"]
    )


    hotwater = st.selectbox(
        "Hot Water Heating",
        ["yes","no"]
    )


    ac = st.selectbox(
        "Air Conditioning",
        ["yes","no"]
    )


    prefarea = st.selectbox(
        "Preferred Area",
        ["yes","no"]
    )



    st.markdown(
    "</div>",
    unsafe_allow_html=True
    )





# ==============================
# PREDICTION
# ==============================


if st.button(
"👑 Evaluate Property"
):


    input_data = pd.DataFrame({

        "area":[area],

        "bedrooms":[bedrooms],

        "bathrooms":[bathrooms],

        "stories":[stories],

        "parking":[parking],

        "guestroom":[guestroom],

        "basement":[basement],

        "hotwaterheating":[hotwater],

        "airconditioning":[ac],

        "prefarea":[prefarea],

        "furnishingstatus":[furnishing]

    })



    prediction = model.predict(
        input_data
    )[0]



    price_sqft = prediction / area



    lower_price = prediction * 0.90

    upper_price = prediction * 1.10



    # CATEGORY


    if prediction < 4000000:


        category="Silver Residence 🏠"


    elif prediction < 7000000:


        category="Gold Residence 🏡"


    elif prediction < 10000000:


        category="Royal Estate 👑"


    else:


        category="Maharaja Palace 💎"




    st.balloons()



    st.markdown(

f"""

<div class="price-card">

<h2>
Estimated Property Value
</h2>


<h1>
₹ {prediction:,.0f}
</h1>


<h2>
₹ {price_sqft:,.0f} / sq.ft
</h2>


</div>

<br>


<div class="badge">

{category}

</div>


""",

unsafe_allow_html=True

)



    st.info(

f"""

### 📊 Estimated Market Range

Minimum:

₹ {lower_price:,.0f}


Maximum:

₹ {upper_price:,.0f}


Variation considered: ±10%

"""

)



    col1,col2,col3 = st.columns(3)


    with col1:

        st.markdown(
        """
        <div class="metric-box">

        <h3>Model</h3>

        XGBoost

        </div>

        """,
        unsafe_allow_html=True
        )


    with col2:


        st.markdown(
        """
        <div class="metric-box">

        <h3>Accuracy</h3>

        R² 79%

        </div>

        """,
        unsafe_allow_html=True
        )


    with col3:


        st.markdown(
        """
        <div class="metric-box">

        <h3>Speed</h3>

        <1 second

        </div>

        """,
        unsafe_allow_html=True
        )



    st.subheader(
        "📋 Property Summary"
    )


    st.dataframe(
        input_data.T,
        width="stretch"
    )




# ==============================
# FOOTER
# ==============================


st.markdown(
"""

<br><br>

<center>

👑 EstateAI Royal Valuation System

<br>

Powered by XGBoost Machine Learning

</center>

""",

unsafe_allow_html=True
)