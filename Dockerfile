FROM python:3.7

RUN apt-get update && apt-get install -y curl cmake

RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/ubuntu/18.04/prod.list > /etc/apt/sources.list.d/mssql-release.list

RUN apt-get install -y unixodbc-dev
RUN apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql17 mssql-tools unixodbc-dev

ENV PATH="${PATH}:/opt/mssql-tools/bin:/opt/mssql-tools/bin"

# Debug
#RUN mkdir app
COPY requirements.txt /app/requirements.txt
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python"]
CMD ["main.py"]