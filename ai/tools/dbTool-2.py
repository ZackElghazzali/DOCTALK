from typing import Dict, Any, Optional
from agent_framework import tool
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, Date, ForeignKey, Time
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime
from request_context import get_verified_user_id
import os

Base = declarative_base()


# SQLAlchemy models — unchanged from AG2 version
class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, unique=True, nullable=False, index=True)
    name = Column(Text, nullable=False)
    dob = Column(Date)
    gender = Column(Text)
    email = Column(Text)
    phone = Column(Text)
    medical_history = relationship("MedicalHistory", back_populates="patient", uselist=False, cascade="all, delete-orphan")
    appointments = relationship("Appointment", back_populates="patient", cascade="all, delete-orphan")
    prescriptions = relationship("Prescription", back_populates="patient", cascade="all, delete-orphan")


class MedicalHistory(Base):
    __tablename__ = "medical_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("patients.user_id"), nullable=False)
    conditions = Column(Text)
    allergies = Column(Text)
    surgeries = Column(Text)
    patient = relationship("Patient", back_populates="medical_history")


class Appointment(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("patients.user_id", ondelete="CASCADE"), nullable=False, index=True)
    date = Column(Date, nullable=False)
    time = Column(Time, nullable=False)
    doctor = Column(Text)
    specialty = Column(Text)
    type = Column(Text)
    status = Column(Text)
    patient = relationship("Patient", back_populates="appointments")


class Prescription(Base):
    __tablename__ = "prescriptions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("patients.user_id", ondelete="CASCADE"), nullable=False, index=True)
    medication = Column(Text)
    dosage = Column(Text)
    frequency = Column(Text)
    prescribing_doctor = Column(Text)
    start_date = Column(Date)
    refills_remaining = Column(Integer, default=0)
    active = Column(Boolean, default=True)
    patient = relationship("Patient", back_populates="prescriptions")


class DatabaseConnection:
    """Manages SQLAlchemy DB connection/session — unchanged from AG2 version."""

    def __init__(self, connection_string: Optional[str] = None):
        if connection_string is None:
            connection_string = os.getenv("DATABASE_URL")
        if not connection_string:
            raise ValueError("DATABASE_URL env var not set")
        self._connection_string = connection_string
        self._engine = create_engine(connection_string, echo=False)
        self._Session = sessionmaker(bind=self._engine)

    def _get_session(self):
        return self._Session()

    def test_connection(self):
        try:
            session = self._get_session()
            count = session.query(Patient).count()
            session.close()
            print(f"DB connection successful. Found {count} patients.")
            return True
        except Exception as e:
            print(f"!!! DB connection failed: {e}")
            return False


def verify_user_access(requested_user_id: int) -> Optional[Dict[str, Any]]:
    verified_user_id = get_verified_user_id()
    if verified_user_id is None:
        return {"error": "Access denied: no authenticated user. Please log in."}
    if requested_user_id != verified_user_id:
        print(f"[SECURITY] Access denied: user {verified_user_id} attempted to access data for user {requested_user_id}")
        return {"error": "Access denied: you can only access your own data."}
    return None


# ─── Tool classes ─────────────────────────────────────────────────────────────
# AG2 pattern per tool: XxxArgs(BaseModel) + XxxTool(BaseTool) with _run/_arun
# MAF pattern: plain class with @tool __call__, constructor injection unchanged

class QueryPatientInfoTool:
    def __init__(self, db_connection: DatabaseConnection):
        self._db_connection = db_connection

    @tool
    def __call__(self, user_id: int) -> Dict[str, Any]:
        """
        Query basic patient information including name, DOB, and contact details.

        Args:
            user_id: Patient user ID.
        """
        access_error = verify_user_access(user_id)
        if access_error:
            return access_error
        print(f"[DB tool] querying patient info for user_id={user_id}")
        session = self._db_connection._get_session()
        try:
            patient = session.query(Patient).filter_by(user_id=user_id).first()
            if not patient:
                return {"error": f"patient with user_id {user_id} not found"}
            return {"name": patient.name, "dob": patient.dob, "email": patient.email, "phone": patient.phone}
        except Exception as e:
            return {"error": f"DB error: {str(e)}"}
        finally:
            session.close()


class QueryMedicalHistoryTool:
    def __init__(self, db_connection: DatabaseConnection):
        self._db_connection = db_connection

    @tool
    def __call__(self, user_id: int) -> Dict[str, Any]:
        """
        Query patient medical history including conditions, allergies, and past surgeries.

        Args:
            user_id: Patient user ID.
        """
        access_error = verify_user_access(user_id)
        if access_error:
            return access_error
        print(f"[DB tool] querying medical history for user_id={user_id}")
        session = self._db_connection._get_session()
        try:
            history = session.query(MedicalHistory).filter_by(user_id=user_id).first()
            if not history:
                return {"error": f"medical history for user_id {user_id} not found"}
            return {
                "conditions": history.conditions or "",
                "allergies": history.allergies or "",
                "surgeries": history.surgeries or "",
            }
        except Exception as e:
            return {"error": f"DB error: {str(e)}"}
        finally:
            session.close()


class QueryAppointmentsTool:
    def __init__(self, db_connection: DatabaseConnection):
        self._db_connection = db_connection

    @tool
    def __call__(self, user_id: int) -> Dict[str, Any]:
        """
        Query all appointments for a patient.

        Args:
            user_id: Patient user ID.
        """
        access_error = verify_user_access(user_id)
        if access_error:
            return access_error
        print(f"[DB tool] querying appointments for user_id={user_id}")
        session = self._db_connection._get_session()
        try:
            appointments = session.query(Appointment).filter_by(user_id=user_id).all()
            if not appointments:
                return {"error": f"no appointments found for user_id {user_id}"}
            appointments_list = [
                {
                    "id": apt.id, "date": apt.date, "time": apt.time,
                    "doctor": apt.doctor, "specialty": apt.specialty,
                    "type": apt.type, "status": apt.status,
                }
                for apt in appointments
            ]
            return {"appointments": appointments_list, "count": len(appointments_list)}
        except Exception as e:
            return {"error": f"DB error: {str(e)}"}
        finally:
            session.close()


class QueryPrescriptionsTool:
    def __init__(self, db_connection: DatabaseConnection):
        self._db_connection = db_connection

    @tool
    def __call__(self, user_id: int, active_only: bool = True) -> Dict[str, Any]:
        """
        Query patient prescriptions. By default returns only active prescriptions.

        Args:
            user_id: Patient user ID.
            active_only: If True, return only active prescriptions (default: True).
        """
        access_error = verify_user_access(user_id)
        if access_error:
            return access_error
        print(f"[DB tool] querying prescriptions for user_id={user_id}, active_only={active_only}")
        session = self._db_connection._get_session()
        try:
            query = session.query(Prescription).filter_by(user_id=user_id)
            if active_only:
                query = query.filter_by(active=True)
            prescriptions = query.all()
            if not prescriptions:
                return {"error": f"no prescriptions found for user_id {user_id}"}
            prescriptions_list = [
                {
                    "id": rx.id, "medication": rx.medication, "dosage": rx.dosage,
                    "frequency": rx.frequency, "prescribing_doctor": rx.prescribing_doctor,
                    "start_date": rx.start_date, "refills_remaining": rx.refills_remaining,
                    "active": rx.active,
                }
                for rx in prescriptions
            ]
            return {"prescriptions": prescriptions_list, "count": len(prescriptions_list)}
        except Exception as e:
            return {"error": f"DB error: {str(e)}"}
        finally:
            session.close()


class UpdatePatientRecordTool:
    IMMUTABLE_FIELDS: Dict[str, set] = {
        "_all": {"user_id"},
        "patients": {"dob"},
    }

    def __init__(self, db_connection: DatabaseConnection, models: Dict[str, Any] = None):
        self._db_connection = db_connection
        self.TABLE_MAP = models or {
            "patients": Patient,
            "medical_history": MedicalHistory,
            "appointments": Appointment,
        }

    @tool
    def __call__(
        self,
        user_id: int,
        table: str,
        updates: Dict[str, Any],
        record_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Update patient records inside the database.
        Can modify records in the patients, appointments, and medical_history tables.

        Args:
            user_id: Patient user ID.
            table: Table to update ("patients", "medical_history", or "appointments").
            updates: Dictionary of field names and their new values.
            record_id: Record ID — required when updating appointments.
        """
        access_error = verify_user_access(user_id)
        if access_error:
            return access_error
        if "user_id" in updates:
            return {"error": "Access denied: cannot modify user_id fields."}
        print(f"[DB write] updating {table} for user_id={user_id}, record_id={record_id}, updates={updates}")
        if table not in self.TABLE_MAP:
            return {"error": f"Invalid table: {table}. Valid tables: {list(self.TABLE_MAP.keys())}"}

        immutable = self.IMMUTABLE_FIELDS["_all"] | self.IMMUTABLE_FIELDS.get(table, set())
        blocked_fields = immutable & updates.keys()
        if blocked_fields:
            print(f"[DB write] warning: attempt to modify immutable fields {blocked_fields} on {table} — skipping")
            updates = {k: v for k, v in updates.items() if k not in immutable}
        if not updates:
            return {"error": f"No updatable fields provided. Fields {blocked_fields} are immutable for {table}."}

        model = self.TABLE_MAP[table]
        session = self._db_connection._get_session()
        try:
            if table in ["appointments"]:
                if record_id is None:
                    return {"error": f"record_id is required for {table} table"}
                record = session.query(model).filter_by(user_id=user_id, id=record_id).first()
            else:
                record = session.query(model).filter_by(user_id=user_id).first()

            if not record:
                return {"error": f"Record not found in {table} for user_id={user_id}" + (f", record_id={record_id}" if record_id else "")}

            updated_fields = []
            for field, value in updates.items():
                if hasattr(record, field):
                    setattr(record, field, value)
                    updated_fields.append(field)
                else:
                    print(f"[DB write] warning: field '{field}' doesn't exist on {table}")

            if not updated_fields:
                return {"error": "no valid fields to update"}

            session.commit()
            return {
                "success": True, "table": table, "user_id": user_id,
                "record_id": record_id, "updated_fields": updated_fields,
                "message": f"Updated {len(updated_fields)} fields in {table}",
            }
        except Exception as e:
            session.rollback()
            return {"error": f"DB error: {str(e)}"}
        finally:
            session.close()


class AddPatientRecordTool:
    def __init__(self, db_connection: DatabaseConnection, models: Dict[str, Any] = None):
        self._db_connection = db_connection
        self.TABLE_MAP = models or {"appointments": Appointment}

    @tool
    def __call__(self, table: str, record_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new record to the appointments table in the database.
        Do NOT include "id" — it is auto-generated upon insertion.
        For appointments: user_id, date (YYYY-MM-DD), time (HH:MM:SS), doctor, specialty, type, status.

        Args:
            table: Table to add the record to.
            record_data: Dictionary containing all fields for the new record.
        """
        verified_user_id = get_verified_user_id()
        if verified_user_id is None:
            return {"error": "Access denied: no authenticated user. Please log in."}

        if "user_id" in record_data:
            if record_data["user_id"] != verified_user_id:
                print(f"[SECURITY] Access denied: User {verified_user_id} attempted to add record for user {record_data['user_id']}")
                return {"error": "Access denied: you can only add records for yourself."}
        else:
            record_data["user_id"] = verified_user_id

        print(f"[DB write] adding record to {table}: {record_data}")
        if table not in self.TABLE_MAP:
            return {"error": f"Invalid table: {table}. Valid tables: {list(self.TABLE_MAP.keys())}"}

        record_data_clean = {k: v for k, v in record_data.items() if k != "id"}
        model = self.TABLE_MAP[table]
        session = self._db_connection._get_session()
        try:
            if table == "appointments":
                if isinstance(record_data_clean.get("date"), str):
                    record_data_clean["date"] = datetime.strptime(record_data_clean["date"], "%Y-%m-%d").date()
                if isinstance(record_data_clean.get("time"), str):
                    time_str = record_data_clean["time"]
                    try:
                        record_data_clean["time"] = datetime.strptime(time_str, "%H:%M:%S").time()
                    except ValueError:
                        record_data_clean["time"] = datetime.strptime(time_str, "%H:%M").time()

            new_record = model(**record_data_clean)
            session.add(new_record)
            session.commit()
            session.refresh(new_record)
            return {
                "success": True, "table": table,
                "record_id": new_record.id,
                "message": f"Added new record to {table} with id={new_record.id}",
            }
        except Exception as e:
            session.rollback()
            return {"error": f"DB error: {str(e)}"}
        finally:
            session.close()


class DeletePatientRecordTool:
    def __init__(self, db_connection: DatabaseConnection, models: Dict[str, Any] = None):
        self._db_connection = db_connection
        self.TABLE_MAP = models or {"appointments": Appointment}

    @tool
    def __call__(self, table: str, record_id: int, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Delete a record from the appointments table in the database.
        Requires both user_id and record_id to identify the specific record.

        Args:
            table: Table to delete from.
            record_id: ID of the record to delete.
            user_id: Patient user ID (optional — defaults to authenticated user).
        """
        verified_user_id = get_verified_user_id()
        if verified_user_id is None:
            return {"error": "Access denied: no authenticated user. Please log in."}
        if user_id is not None and user_id != verified_user_id:
            print(f"[SECURITY] Access denied: user {verified_user_id} attempted to delete record for user {user_id}")
            return {"error": "Access denied: you can only delete your own records."}

        user_id = verified_user_id
        print(f"[DB write] deleting from {table}: user_id={user_id}, record_id={record_id}")

        if table not in self.TABLE_MAP:
            return {"error": f"Invalid table: {table}. Valid tables: {list(self.TABLE_MAP.keys())}"}

        model = self.TABLE_MAP[table]
        session = self._db_connection._get_session()
        try:
            if table == "patients":
                record = session.query(model).filter_by(user_id=user_id).first()
                if record and record.id != record_id:
                    return {"error": "Access denied: Record ID does not match your patient record."}
            else:
                record = session.query(model).filter_by(user_id=user_id, id=record_id).first()

            if not record:
                return {"error": f"Record not found in {table} for user_id={user_id}, record_id={record_id}"}

            session.delete(record)
            session.commit()
            return {
                "success": True, "table": table,
                "record_id": record_id,
                "message": f"Deleted record {record_id} from {table}",
            }
        except Exception as e:
            session.rollback()
            return {"error": f"DB error: {str(e)}"}
        finally:
            session.close()
