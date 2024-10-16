from typing import Optional, Tuple

from chromadb.auth import UserIdentity
from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT


def _singleton_tenant_database_if_applicable(
    user_identity: UserIdentity,
    overwrite_singleton_tenant_database_access_from_auth: bool,
) -> Tuple[Optional[str], Optional[str]]:
    """
    If settings.chroma_overwrite_singleton_tenant_database_access_from_auth
    is False, this function always returns (None, None).

    If settings.chroma_overwrite_singleton_tenant_database_access_from_auth
    is True, follows the following logic:
    - If the user only has access to a single tenant, this function will
        return that tenant as its first return value.
    - If the user only has access to a single database, this function will
        return that database as its second return value. If the user has
        access to multiple tenants and/or databases, including "*", this
        function will return None for the corresponding value(s).
    - If the user has access to multiple tenants and/or databases this
        function will return None for the corresponding value(s).
    """
    if not overwrite_singleton_tenant_database_access_from_auth:
        return None, None
    tenant = None
    database = None
    user_tenant = user_identity.tenant
    user_databases = user_identity.databases
    if user_tenant and user_tenant != "*":
        tenant = user_tenant
    if user_databases and len(user_databases) == 1 and user_databases[0] != "*":
        database = user_databases[0]
    return tenant, database


def maybe_set_tenant_and_database(
    user_identity: UserIdentity,
    overwrite_singleton_tenant_database_access_from_auth: bool,
    tenant: Optional[str] = None,
    database: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    (
        new_tenant,
        new_database,
    ) = _singleton_tenant_database_if_applicable(
        user_identity=user_identity,
        overwrite_singleton_tenant_database_access_from_auth=overwrite_singleton_tenant_database_access_from_auth,
    )

    # The only error case is if the user provides a tenant and databased that
    # don't match what we resolved from auth. This can incorrectly happen when
    # there is no auth provider set, but overwrite_singleton_tenant_database_access_from_auth
    # is set to True. In this case, we'll resolve tenant/database to the default
    # values, which might not match the provided values. Thus, it's important
    # to ensure that the flag is set to True only when there is an auth provider.
    if tenant and tenant != DEFAULT_TENANT and new_tenant and new_tenant != tenant:
        raise ValueError(
            f"Resolved tenant {new_tenant} doesn't match provided tenant {tenant}."
        )
    if (
        database
        and database != DEFAULT_DATABASE
        and new_database
        and new_database != database
    ):
        raise ValueError(
            f"Resolved database {new_database} doesn't match provided database {database}."
        )

    if (not tenant or tenant == DEFAULT_TENANT) and new_tenant:
        tenant = new_tenant
    if (not database or database == DEFAULT_DATABASE) and new_database:
        database = new_database

    return tenant, database
